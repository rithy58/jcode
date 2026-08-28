//! Streamable HTTP transport for MCP servers (spec revision 2025-03-26).
//!
//! Speaks JSON-RPC over a single HTTP endpoint: every client message is an
//! HTTP POST whose response is either a direct `application/json` body or a
//! `text/event-stream` that eventually carries the matching JSON-RPC
//! response. Session continuity uses the `Mcp-Session-Id` header the server
//! assigns during `initialize`; the negotiated protocol version is echoed on
//! subsequent requests via `MCP-Protocol-Version`.

use super::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const SESSION_HEADER: &str = "mcp-session-id";
const PROTOCOL_VERSION_HEADER: &str = "mcp-protocol-version";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Shared connection state for one streamable HTTP MCP server.
///
/// Unlike stdio there is no child process: each request is an independent
/// HTTP round-trip, correlated to a session by `Mcp-Session-Id`.
pub struct HttpTransport {
    name: String,
    url: String,
    headers: HashMap<String, String>,
    client: reqwest::Client,
    request_id: AtomicU64,
    session_id: RwLock<Option<String>>,
    protocol_version: RwLock<Option<String>>,
}

impl HttpTransport {
    pub fn new(name: String, url: String, headers: HashMap<String, String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .context("Failed to build HTTP client for MCP transport")?;
        Ok(Self {
            name,
            url,
            headers,
            client,
            request_id: AtomicU64::new(1),
            session_id: RwLock::new(None),
            protocol_version: RwLock::new(None),
        })
    }

    /// Record the protocol version negotiated during `initialize` so it is
    /// echoed on every subsequent request, as the spec requires.
    pub fn set_protocol_version(&self, version: &str) {
        *self
            .protocol_version
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(version.to_string());
    }

    fn session_id(&self) -> Option<String> {
        self.session_id
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Send a JSON-RPC request and wait for the matching response, whether the
    /// server answers with a plain JSON body or an SSE stream.
    pub async fn request(&self, method: &str, params: Option<Value>) -> Result<JsonRpcResponse> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let body = serde_json::to_string(&request)?;
        let response = self.post(body).await?;
        self.extract_response(response, id).await
    }

    /// Send a JSON-RPC notification. Servers acknowledge with 202 Accepted
    /// (some reply 200); any success status counts, the body is ignored.
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<()> {
        let notification = JsonRpcNotification::new(method, params);
        let body = serde_json::to_string(&notification)?;
        let _ = self.post(body).await?;
        Ok(())
    }

    /// Best-effort session teardown (HTTP DELETE with the session id). Servers
    /// may respond 405 when they do not support explicit termination.
    pub async fn close(&self) {
        let Some(session_id) = self.session_id() else {
            return;
        };
        let mut request = self
            .client
            .delete(&self.url)
            .header(SESSION_HEADER, session_id);
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), request.send()).await;
    }

    async fn post(&self, body: String) -> Result<reqwest::Response> {
        let mut request = self
            .client
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        if let Some(session_id) = self.session_id() {
            request = request.header(SESSION_HEADER, session_id);
        }
        if let Some(version) = self
            .protocol_version
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            request = request.header(PROTOCOL_VERSION_HEADER, version);
        }

        let response = request.body(body).send().await.with_context(|| {
            format!(
                "Failed to reach MCP HTTP server '{}' at {}",
                self.name, self.url
            )
        })?;

        // The server assigns the session id in its `initialize` response; keep
        // whatever the latest response carries.
        if let Some(session_id) = response
            .headers()
            .get(SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            *self
                .session_id
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(session_id.to_string());
        }

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "MCP HTTP server '{}' returned {}: {}",
                self.name,
                status,
                truncate_for_error(&body)
            );
        }

        Ok(response)
    }

    async fn extract_response(
        &self,
        response: reqwest::Response,
        id: u64,
    ) -> Result<JsonRpcResponse> {
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();

        if content_type.starts_with("text/event-stream") {
            return self.read_sse_response(response, id).await;
        }

        let bytes = response.bytes().await.with_context(|| {
            format!("Failed to read response body from MCP server '{}'", self.name)
        })?;
        serde_json::from_slice::<JsonRpcResponse>(&bytes).with_context(|| {
            format!(
                "MCP HTTP server '{}' returned a non-JSON-RPC body: {}",
                self.name,
                truncate_for_error(&String::from_utf8_lossy(&bytes))
            )
        })
    }

    /// Read an SSE stream until the JSON-RPC response for `id` arrives.
    /// Server-initiated notifications/requests on the stream are skipped.
    async fn read_sse_response(
        &self,
        response: reqwest::Response,
        id: u64,
    ) -> Result<JsonRpcResponse> {
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::default();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| {
                format!("Failed to read SSE stream from MCP server '{}'", self.name)
            })?;
            for data in parser.push(&chunk) {
                let Ok(message) = serde_json::from_str::<JsonRpcResponse>(&data) else {
                    continue;
                };
                if message.id == Some(id) && (message.result.is_some() || message.error.is_some()) {
                    return Ok(message);
                }
            }
        }
        bail!(
            "MCP HTTP server '{}' closed the SSE stream without answering request {}",
            self.name,
            id
        )
    }
}

/// Incremental Server-Sent Events parser. Feed raw bytes, get back the joined
/// `data:` payload of each completed event. `event:`/`id:`/`retry:` fields and
/// comment lines are ignored; multi-line data is joined with newlines per the
/// SSE spec. Splitting on byte boundaries is safe because UTF-8 continuation
/// bytes can never equal `\n`.
#[derive(Default)]
struct SseParser {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let raw_line: Vec<u8> = self.buffer.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&raw_line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    events.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data_lines
                    .push(data.strip_prefix(' ').unwrap_or(data).to_string());
            }
        }
        events
    }
}

fn truncate_for_error(text: &str) -> String {
    const MAX_CHARS: usize = 300;
    let mut truncated: String = text.chars().take(MAX_CHARS).collect();
    if truncated.len() < text.len() {
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod http_tests;
