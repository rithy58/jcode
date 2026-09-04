//! MCP Client - handles communication with a single MCP server
//!
//! Supports two transports: stdio (a spawned child process speaking JSON-RPC
//! over stdin/stdout) and streamable HTTP (JSON-RPC over HTTP POST, see
//! `super::http`). The transport is chosen from the server's config.

use super::http::HttpTransport;
use super::protocol::*;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};

const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Protocol revision requested over stdio (the original transport).
const STDIO_PROTOCOL_VERSION: &str = "2024-11-05";
/// Protocol revision that introduced streamable HTTP.
const HTTP_PROTOCOL_VERSION: &str = "2025-03-26";

/// Transport-specific half of a connection.
#[derive(Clone)]
enum HandleTransport {
    Stdio(StdioHandle),
    Http(Arc<HttpTransport>),
}

#[derive(Clone)]
struct StdioHandle {
    request_id: Arc<AtomicU64>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    writer_tx: mpsc::Sender<String>,
}

/// Shared communication handle for an MCP server.
/// Multiple sessions can hold clones of this and send concurrent requests.
/// Request/response correlation by ID ensures no interference.
#[derive(Clone)]
pub struct McpHandle {
    pub(crate) name: String,
    transport: HandleTransport,
    server_info: Arc<std::sync::RwLock<Option<ServerInfo>>>,
    capabilities: Arc<std::sync::RwLock<ServerCapabilities>>,
    tools: Arc<std::sync::RwLock<Vec<McpToolDef>>>,
}

impl McpHandle {
    /// Send a request and wait for response
    pub async fn request(&self, method: &str, params: Option<Value>) -> Result<JsonRpcResponse> {
        let response = match &self.transport {
            HandleTransport::Stdio(stdio) => {
                let id = stdio.request_id.fetch_add(1, Ordering::SeqCst);
                let request = JsonRpcRequest::new(id, method, params);

                let (tx, rx) = oneshot::channel();
                {
                    let mut pending = stdio.pending.lock().await;
                    pending.insert(id, tx);
                }

                let msg = serde_json::to_string(&request)? + "\n";
                stdio
                    .writer_tx
                    .send(msg)
                    .await
                    .context("Failed to send request")?;

                tokio::time::timeout(REQUEST_TIMEOUT, rx)
                    .await
                    .context("Request timeout")?
                    .context("Channel closed")?
            }
            HandleTransport::Http(http) => tokio::time::timeout(
                REQUEST_TIMEOUT,
                http.request(method, params),
            )
            .await
            .context("Request timeout")??,
        };

        if let Some(err) = &response.error {
            anyhow::bail!("MCP error {}: {}", err.code, err.message);
        }

        Ok(response)
    }

    /// Send a notification (no response expected)
    async fn notify(&self, method: &str, params: Option<Value>) -> Result<()> {
        match &self.transport {
            HandleTransport::Stdio(stdio) => {
                let notif = JsonRpcNotification::new(method, params);
                let msg = serde_json::to_string(&notif)? + "\n";
                stdio.writer_tx.send(msg).await?;
            }
            HandleTransport::Http(http) => {
                http.notify(method, params).await?;
            }
        }
        Ok(())
    }

    /// Call a tool
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolCallResult> {
        let arguments = if arguments.is_null() {
            Value::Object(serde_json::Map::new())
        } else {
            arguments
        };
        let params = ToolCallParams {
            name: name.to_string(),
            arguments,
        };

        let response = self
            .request("tools/call", Some(serde_json::to_value(params)?))
            .await?;

        let result = response.result.context("No result from tool call")?;
        let tool_result: ToolCallResult = serde_json::from_value(result)?;

        Ok(tool_result)
    }

    /// Get the server name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get server info
    pub fn server_info(&self) -> Option<ServerInfo> {
        self.server_info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Get available tools
    pub fn tools(&self) -> Vec<McpToolDef> {
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Refresh the list of available tools
    pub async fn refresh_tools(&self) -> Result<()> {
        let response = self.request("tools/list", None).await?;

        if let Some(result) = response.result {
            let tools_result: ToolsListResult = serde_json::from_value(result)?;
            *self
                .tools
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = tools_result.tools;
        }

        Ok(())
    }
}

/// MCP Client - owns the connection (child process for stdio, HTTP session
/// for streamable HTTP) and provides shared handles. Only one McpClient
/// exists per MCP server, but many McpHandle clones can be distributed to
/// different sessions.
pub struct McpClient {
    handle: McpHandle,
    /// Child process for stdio servers. `None` for HTTP servers.
    child: Option<Child>,
}

impl McpClient {
    /// Connect to an MCP server, inheriting the current process working directory
    pub async fn connect(name: String, config: &McpServerConfig) -> Result<Self> {
        Self::connect_in_dir(name, config, None).await
    }

    /// Connect to an MCP server, optionally running it in `working_dir`.
    ///
    /// HTTP servers connect over the network and ignore `working_dir`. For
    /// stdio servers the working directory is only applied when it exists;
    /// otherwise the subprocess falls back to inheriting the current process
    /// cwd (issue #557).
    pub async fn connect_in_dir(
        name: String,
        config: &McpServerConfig,
        working_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
        if config.is_http() {
            return Self::connect_http(name, config).await;
        }
        let working_dir = working_dir.filter(|dir| dir.is_dir());
        crate::logging::info(&format!(
            "MCP: Connecting to '{}' ({} {:?}) cwd={:?}",
            name, config.command, config.args, working_dir
        ));

        // Credentials must be opted into an MCP server explicitly through its
        // config. The long-lived jcode daemon contains provider credentials in
        // its process environment, and blindly inheriting them exposes those
        // credentials to every configured MCP executable (issue #771).
        let inherited: HashMap<String, String> = std::env::vars().collect();
        let env = mcp_child_env(inherited, &config.env);

        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = working_dir {
            command.current_dir(dir);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {}", config.command))?;

        let stdin = child.stdin.take().context("No stdin")?;
        let stdout = child.stdout.take().context("No stdout")?;
        let stderr = child.stderr.take().context("No stderr")?;

        // Spawn stderr reader
        let server_name = name.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            crate::logging::warn(&format!(
                                "MCP [{}] stderr: {}",
                                server_name, trimmed
                            ));
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // Setup channels
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (writer_tx, mut writer_rx) = mpsc::channel::<String>(32);

        // Spawn writer task
        let mut stdin = stdin;
        tokio::spawn(async move {
            while let Some(msg) = writer_rx.recv().await {
                if stdin.write_all(msg.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        // Spawn reader task
        let pending_clone = Arc::clone(&pending);
        let reader_name = name.clone();
        let mut reader = BufReader::new(stdout);
        tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        crate::logging::debug(&format!("MCP [{}]: stdout EOF", reader_name));
                        break;
                    }
                    Ok(_) => {
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            if let Some(id) = response.id {
                                let mut pending = pending_clone.lock().await;
                                if let Some(tx) = pending.remove(&id) {
                                    let _ = tx.send(response);
                                }
                            }
                        } else {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                crate::logging::debug(&format!(
                                    "MCP [{}] non-JSON output: {}",
                                    reader_name, trimmed
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        crate::logging::warn(&format!("MCP [{}] read error: {}", reader_name, e));
                        break;
                    }
                }
            }
        });

        let handle = McpHandle {
            name: name.clone(),
            transport: HandleTransport::Stdio(StdioHandle {
                request_id: Arc::new(AtomicU64::new(1)),
                pending,
                writer_tx,
            }),
            server_info: Arc::new(std::sync::RwLock::new(None)),
            capabilities: Arc::new(std::sync::RwLock::new(ServerCapabilities::default())),
            tools: Arc::new(std::sync::RwLock::new(Vec::new())),
        };

        let mut client = Self {
            handle,
            child: Some(child),
        };
        client.finish_connect(&name).await?;
        Ok(client)
    }

    /// Connect to a streamable HTTP MCP server (issue: HTTP/SSE entries used
    /// to be recognized but skipped).
    pub(crate) async fn connect_http(name: String, config: &McpServerConfig) -> Result<Self> {
        let url = config
            .url
            .as_deref()
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .with_context(|| format!("MCP HTTP server '{}' has no 'url' configured", name))?
            .to_string();

        crate::logging::info(&format!("MCP: Connecting to '{}' over HTTP ({})", name, url));

        let transport = HttpTransport::new(name.clone(), url, config.headers.clone())?;
        let handle = McpHandle {
            name: name.clone(),
            transport: HandleTransport::Http(Arc::new(transport)),
            server_info: Arc::new(std::sync::RwLock::new(None)),
            capabilities: Arc::new(std::sync::RwLock::new(ServerCapabilities::default())),
            tools: Arc::new(std::sync::RwLock::new(Vec::new())),
        };

        let mut client = Self {
            handle,
            child: None,
        };
        client.finish_connect(&name).await?;
        Ok(client)
    }

    /// Shared tail of both connect paths: initialize handshake + first
    /// tools/list.
    async fn finish_connect(&mut self, name: &str) -> Result<()> {
        self.initialize()
            .await
            .with_context(|| format!("MCP server '{}' failed to initialize", name))?;

        self.handle
            .refresh_tools()
            .await
            .with_context(|| format!("MCP server '{}' failed to list tools", name))?;

        crate::logging::info(&format!(
            "MCP: Connected to '{}' with {} tools",
            name,
            self.handle.tools().len()
        ));
        Ok(())
    }

    /// Get a shareable handle to this client
    pub fn handle(&self) -> McpHandle {
        self.handle.clone()
    }

    /// Initialize the MCP connection
    async fn initialize(&mut self) -> Result<()> {
        let requested_version = match &self.handle.transport {
            HandleTransport::Stdio(_) => STDIO_PROTOCOL_VERSION,
            HandleTransport::Http(_) => HTTP_PROTOCOL_VERSION,
        };
        let params = InitializeParams {
            protocol_version: requested_version.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "jcode".to_string(),
                version: jcode_build_meta::pkg_version().to_string(),
            },
        };

        let response = self
            .handle
            .request("initialize", Some(serde_json::to_value(params)?))
            .await?;

        if let Some(result) = response.result {
            let init_result: InitializeResult = serde_json::from_value(result)?;
            // Streamable HTTP echoes the negotiated version on every
            // subsequent request via the MCP-Protocol-Version header.
            if let HandleTransport::Http(http) = &self.handle.transport {
                http.set_protocol_version(&init_result.protocol_version);
            }
            *self
                .handle
                .server_info
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = init_result.server_info;
            *self
                .handle
                .capabilities
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = init_result.capabilities;
        }

        // Send initialized notification
        self.handle
            .notify("notifications/initialized", None)
            .await?;

        Ok(())
    }

    /// Check if server is still running. HTTP servers have no local process;
    /// they count as running until a request fails.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => true,
        }
    }

    /// Shutdown the server
    pub async fn shutdown(&mut self) {
        match &self.handle.transport {
            HandleTransport::Stdio(stdio) => {
                let _ = stdio
                    .writer_tx
                    .send("{\"jsonrpc\":\"2.0\",\"method\":\"shutdown\"}\n".to_string())
                    .await;

                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                if let Some(child) = &mut self.child {
                    let _ = child.kill().await;
                }
            }
            HandleTransport::Http(http) => {
                http.close().await;
            }
        }
    }

    // === Legacy compatibility methods that delegate to handle ===

    pub fn name(&self) -> &str {
        &self.handle.name
    }

    pub fn server_info(&self) -> Option<ServerInfo> {
        self.handle.server_info()
    }

    pub fn tools(&self) -> Vec<McpToolDef> {
        self.handle.tools()
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolCallResult> {
        self.handle.call_tool(name, arguments).await
    }

    pub async fn refresh_tools(&self) -> Result<()> {
        self.handle.refresh_tools().await
    }
}

/// Secrets that an MCP child must not receive merely because jcode has them.
///
/// This intentionally applies only to inherited values. A server can still be
/// given any of these names through `McpServerConfig::env`.
fn is_sensitive_inherited_env_key(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    key.ends_with("_API_KEY")
        || key.ends_with("_ACCESS_TOKEN")
        || key.ends_with("_AUTH_TOKEN")
        || matches!(
            key.as_str(),
            "AWS_ACCESS_KEY_ID"
                | "AWS_SECRET_ACCESS_KEY"
                | "AWS_SESSION_TOKEN"
                | "AZURE_CLIENT_SECRET"
                | "GOOGLE_APPLICATION_CREDENTIALS"
        )
}

fn mcp_child_env(
    mut inherited: HashMap<String, String>,
    explicit: &HashMap<String, String>,
) -> HashMap<String, String> {
    inherited.retain(|key, _| !is_sensitive_inherited_env_key(key));
    inherited.extend(explicit.clone());
    inherited
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.start_kill();
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::{McpClient, is_sensitive_inherited_env_key, mcp_child_env};
    use crate::mcp::protocol::McpServerConfig;
    use std::collections::HashMap;

    #[test]
    fn inherited_mcp_env_scrubs_provider_credentials() {
        for key in [
            "ANTHROPIC_API_KEY",
            "openai_api_key",
            "CURSOR_ACCESS_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ] {
            assert!(is_sensitive_inherited_env_key(key), "must scrub {key}");
        }
        for key in ["PATH", "HOME", "RUST_LOG", "JCODE_OPENROUTER_API_KEY_NAME"] {
            assert!(!is_sensitive_inherited_env_key(key), "must preserve {key}");
        }
    }

    #[test]
    fn explicit_mcp_env_can_opt_a_credential_back_in() {
        let inherited = HashMap::from([
            ("PATH".to_string(), "/bin".to_string()),
            ("ANTHROPIC_API_KEY".to_string(), "daemon-secret".to_string()),
        ]);
        let explicit = HashMap::from([(
            "ANTHROPIC_API_KEY".to_string(),
            "server-specific-secret".to_string(),
        )]);

        let env = mcp_child_env(inherited, &explicit);
        assert_eq!(env.get("PATH").map(String::as_str), Some("/bin"));
        assert_eq!(
            env.get("ANTHROPIC_API_KEY").map(String::as_str),
            Some("server-specific-secret")
        );
    }

    /// A minimal fake stdio MCP server (shell script) that reports its own
    /// process cwd as the serverInfo name.
    fn fake_server_config() -> McpServerConfig {
        let script = r#"
while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*)
      printf '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"%s","version":"0"}}}\n' "$PWD"
      ;;
    *'"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}\n'
      ;;
  esac
done
"#;
        McpServerConfig {
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: Default::default(),
            shared: false,
            transport: None,
            url: None,
            headers: std::collections::HashMap::new(),
            enabled: None,
            disabled: None,
        }
    }

    #[tokio::test]
    async fn connect_in_dir_sets_subprocess_cwd() {
        // Issue #557: owned MCP servers must run in the session project dir.
        let dir = tempfile::tempdir().expect("tempdir");
        let expected = dir.path().canonicalize().expect("canonicalize");

        let client = McpClient::connect_in_dir(
            "cwd-test".to_string(),
            &fake_server_config(),
            Some(dir.path()),
        )
        .await
        .expect("connect");

        let reported = client.server_info().expect("server info").name;
        assert_eq!(
            std::path::Path::new(&reported)
                .canonicalize()
                .expect("canonicalize reported"),
            expected
        );
    }

    #[tokio::test]
    async fn connect_in_dir_missing_dir_falls_back_to_inherited_cwd() {
        let client = McpClient::connect_in_dir(
            "cwd-fallback-test".to_string(),
            &fake_server_config(),
            Some(std::path::Path::new("/nonexistent/jcode-557")),
        )
        .await
        .expect("connect should fall back to inherited cwd");

        let reported = client.server_info().expect("server info").name;
        assert!(!reported.is_empty());
    }
}
