use super::SseParser;
use crate::mcp::client::McpClient;
use crate::mcp::protocol::{ContentBlock, McpServerConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[test]
fn sse_parser_joins_multi_line_data_and_ignores_other_fields() {
    let mut parser = SseParser::default();
    let events = parser.push(b"event: message\r\ndata: line one\r\ndata: line two\r\n\r\n");
    assert_eq!(events, vec!["line one\nline two".to_string()]);
}

#[test]
fn sse_parser_handles_data_split_across_chunks() {
    let mut parser = SseParser::default();
    assert!(parser.push(b"data: {\"jsonrpc\":").is_empty());
    assert!(parser.push(b"\"2.0\"}\n").is_empty());
    let events = parser.push(b"\n");
    assert_eq!(events, vec!["{\"jsonrpc\":\"2.0\"}".to_string()]);
}

#[test]
fn sse_parser_emits_multiple_events_from_one_chunk() {
    let mut parser = SseParser::default();
    let events = parser.push(b": comment\ndata: first\n\ndata: second\n\n");
    assert_eq!(events, vec!["first".to_string(), "second".to_string()]);
}

/// Read one HTTP/1.1 request (header block + content-length body) from the
/// stream. Returns None on EOF.
async fn read_http_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&buf[..pos]).to_string();
            let content_length = header
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            let mut body = buf[pos + 4..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut tmp).await.ok()?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            return Some((header, String::from_utf8_lossy(&body).to_string()));
        }
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn json_response(body: &serde_json::Value) -> String {
    let body = body.to_string();
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nmcp-session-id: fake-session-1\r\ncontent-length: {}\r\n\r\n{}",
        body.len(),
        body
    )
}

fn sse_response(events: &[serde_json::Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("event: message\n");
        body.push_str(&format!("data: {}\n\n", event));
    }
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nmcp-session-id: fake-session-1\r\ncontent-length: {}\r\n\r\n{}",
        body.len(),
        body
    )
}

/// A minimal streamable-HTTP MCP server: answers initialize and tools/list
/// with JSON bodies, and tools/call with an SSE stream that emits a server
/// notification before the actual response (both spec-legal shapes).
async fn run_fake_server(listener: TcpListener, seen_headers: Arc<Mutex<Vec<String>>>) {
    loop {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let seen_headers = Arc::clone(&seen_headers);
        tokio::spawn(async move {
            while let Some((header, body)) = read_http_request(&mut stream).await {
                seen_headers
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(header.clone());
                let message: serde_json::Value =
                    serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
                let method = message.get("method").and_then(|m| m.as_str()).unwrap_or("");
                let id = message.get("id").and_then(|i| i.as_u64()).unwrap_or(0);
                let response = match method {
                    "initialize" => json_response(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-http", "version": "1.0.0"}
                        }
                    })),
                    "tools/list" => json_response(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "tools": [{
                                "name": "echo",
                                "description": "Echo a message",
                                "inputSchema": {"type": "object"}
                            }]
                        }
                    })),
                    "tools/call" => sse_response(&[
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "method": "notifications/message",
                            "params": {"level": "info", "data": "working"}
                        }),
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{"type": "text", "text": "echoed over http"}],
                                "isError": false
                            }
                        }),
                    ]),
                    // notifications/initialized and session DELETE land here.
                    _ => "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n".to_string(),
                };
                if stream.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
            }
        });
    }
}

fn http_config(url: String) -> McpServerConfig {
    McpServerConfig {
        command: String::new(),
        args: vec![],
        env: HashMap::new(),
        shared: true,
        transport: Some("http".to_string()),
        url: Some(url),
        headers: HashMap::from([("Authorization".to_string(), "Bearer test-token".to_string())]),
        enabled: None,
        disabled: None,
    }
}

#[tokio::test]
async fn http_transport_end_to_end_initialize_list_and_call() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}/mcp", listener.local_addr().expect("addr"));
    let seen_headers = Arc::new(Mutex::new(Vec::new()));
    let server = tokio::spawn(run_fake_server(listener, Arc::clone(&seen_headers)));

    let client = McpClient::connect("fake-http".to_string(), &http_config(url))
        .await
        .expect("HTTP MCP server must connect");

    assert_eq!(client.server_info().expect("server info").name, "fake-http");
    let tools = client.tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    // tools/call is answered over SSE with a notification before the response.
    let result = client
        .call_tool("echo", serde_json::json!({"message": "hi"}))
        .await
        .expect("tools/call over SSE");
    assert!(!result.is_error);
    match &result.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "echoed over http"),
        other => panic!("expected text content, got {other:?}"),
    }

    server.abort();

    let seen = seen_headers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let initialize_headers = &seen[0];
    assert!(
        initialize_headers
            .to_ascii_lowercase()
            .contains("authorization: bearer test-token"),
        "configured headers must be sent: {initialize_headers}"
    );
    let call_headers = seen.last().expect("tools/call request headers");
    let call_headers = call_headers.to_ascii_lowercase();
    assert!(
        call_headers.contains("mcp-session-id: fake-session-1"),
        "session id from initialize must be echoed on later requests: {call_headers}"
    );
    assert!(
        call_headers.contains("mcp-protocol-version: 2025-03-26"),
        "negotiated protocol version must be echoed on later requests: {call_headers}"
    );
}

#[tokio::test]
async fn http_connect_fails_cleanly_on_server_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}/mcp", listener.local_addr().expect("addr"));
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                while read_http_request(&mut stream).await.is_some() {
                    let body = "internal error";
                    let response = format!(
                        "HTTP/1.1 500 Internal Server Error\r\ncontent-length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    if stream.write_all(response.as_bytes()).await.is_err() {
                        return;
                    }
                }
            });
        }
    });

    let error = match McpClient::connect("failing-http".to_string(), &http_config(url)).await {
        Ok(_) => panic!("a 500 during initialize must fail the connect"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("failing-http") && message.contains("500"),
        "error must name the server and status: {message}"
    );
    server.abort();
}

#[tokio::test]
async fn http_config_without_url_fails_with_clear_error() {
    let mut config = http_config(String::new());
    config.url = None;
    // Force the http path despite the missing URL.
    config.transport = Some("http".to_string());
    let error = match McpClient::connect_http("no-url".to_string(), &config).await {
        Ok(_) => panic!("missing url must fail"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("no 'url' configured"));
}

/// Live acceptance check against a public streamable HTTP MCP server.
/// Ignored by default (network); run explicitly with:
/// `cargo test -p jcode-base --lib mcp::http -- --ignored`
#[tokio::test]
#[ignore = "requires network access to mcp.deepwiki.com"]
async fn live_public_http_server_initialize_and_list_tools() {
    let mut config = http_config("https://mcp.deepwiki.com/mcp".to_string());
    config.headers.clear();

    let client = McpClient::connect("deepwiki".to_string(), &config)
        .await
        .expect("public server must connect");
    assert_eq!(client.server_info().expect("server info").name, "DeepWiki");
    assert!(
        client.tools().iter().any(|tool| tool.name == "ask_question"),
        "tools/list must include ask_question: {:?}",
        client
            .tools()
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>()
    );
}
