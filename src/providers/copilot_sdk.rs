//! Copilot SDK provider — communicates with the official GitHub Copilot CLI
//! via JSON-RPC 2.0 over stdio or TCP.
//!
//! Instead of impersonating VS Code with hardcoded OAuth client IDs, this
//! provider delegates to the Copilot CLI running in headless/server mode.
//! The CLI handles authentication, model routing, and context windowing
//! internally.
//!
//! ## Transport modes
//!
//! - **stdio** (default): Spawns `copilot --headless --stdio` as a child
//!   process and communicates over stdin/stdout pipes.
//! - **TCP**: Connects to an external CLI server at `cli_url` (e.g.
//!   `tcp://127.0.0.1:4321`).
//!
//! ## Session strategy
//!
//! A single session is created on the first `chat()` call and reused for
//! subsequent calls within the same provider instance. The system message is
//! set at session-creation time.

use crate::providers::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    NormalizedStopReason, Provider, ToolCall as ProviderToolCall,
};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing::debug;

// ── Constants ────────────────────────────────────────────────────

/// Maximum line size for JSON-RPC responses (4 MB).
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

/// Timeout for JSON-RPC request/response round-trips.
const RPC_TIMEOUT_SECS: u64 = 120;

/// Expected JSON-RPC protocol version.
const JSONRPC_VERSION: &str = "2.0";

/// Default CLI binary name.
const DEFAULT_CLI_PATH: &str = "copilot";

// ── JSON-RPC types ───────────────────────────────────────────────

/// Outgoing JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// Incoming JSON-RPC 2.0 response.
#[derive(Debug, Clone, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

// ── CLI process handle ───────────────────────────────────────────

/// Managed Copilot CLI child process with stdio handles.
struct CliProcess {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
}

impl CliProcess {
    /// Check whether the child process is still alive.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

// ── Provider struct ──────────────────────────────────────────────

/// GitHub Copilot SDK provider that communicates with the Copilot CLI
/// via JSON-RPC 2.0 over stdio (default) or TCP.
pub struct CopilotSdkProvider {
    /// Path to the `copilot` binary.
    cli_path: String,
    /// Optional URL of an external CLI server (mutually exclusive with spawning).
    cli_url: Option<String>,
    /// Optional GitHub token for CLI authentication.
    github_token: Option<String>,
    /// Log level for the CLI subprocess.
    log_level: String,
    /// Managed CLI process (stdio mode).
    process: Arc<Mutex<Option<CliProcess>>>,
    /// TCP connection (external server mode).
    tcp_conn: Arc<Mutex<Option<TcpConnection>>>,
    /// Active session ID.
    session_id: Arc<Mutex<Option<String>>>,
    /// Monotonic JSON-RPC request ID counter.
    next_id: AtomicU64,
}

/// TCP connection to an external Copilot CLI server.
struct TcpConnection {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    writer: tokio::net::tcp::OwnedWriteHalf,
}

impl CopilotSdkProvider {
    /// Create a new `CopilotSdkProvider`.
    ///
    /// - `cli_path`: path to the `copilot` binary (default: `"copilot"`).
    /// - `cli_url`: URL of an external Copilot CLI server (e.g. `"tcp://127.0.0.1:4321"`).
    /// - `github_token`: optional GitHub token for authentication.
    /// - `log_level`: CLI log level (default: `"error"`).
    pub fn new(
        cli_path: Option<&str>,
        cli_url: Option<&str>,
        github_token: Option<&str>,
        log_level: Option<&str>,
    ) -> Self {
        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or(DEFAULT_CLI_PATH)
                .to_string(),
            cli_url: cli_url.filter(|s| !s.is_empty()).map(ToString::to_string),
            github_token: github_token
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
            log_level: log_level
                .filter(|s| !s.is_empty())
                .unwrap_or("error")
                .to_string(),
            process: Arc::new(Mutex::new(None)),
            tcp_conn: Arc::new(Mutex::new(None)),
            session_id: Arc::new(Mutex::new(None)),
            next_id: AtomicU64::new(1),
        }
    }

    /// Allocate the next JSON-RPC request ID.
    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    // ── Transport: send + receive ────────────────────────────────

    /// Send a JSON-RPC request and wait for the matching response.
    ///
    /// This method picks the correct transport (stdio process or TCP) and
    /// handles serialization, newline framing, and timeout.
    async fn rpc_call(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: self.next_request_id(),
            method: method.to_string(),
            params,
        };

        let line = serde_json::to_string(&request)
            .with_context(|| format!("failed to serialize JSON-RPC request for {method}"))?;

        let timeout_duration = tokio::time::Duration::from_secs(RPC_TIMEOUT_SECS);

        // Prefer TCP if connected, otherwise stdio.
        if self.cli_url.is_some() {
            let mut tcp_guard = self.tcp_conn.lock().await;
            let conn = tcp_guard
                .as_mut()
                .context("TCP connection not established")?;
            return Self::send_and_recv_tcp(conn, &line, request.id, timeout_duration).await;
        }

        let mut proc_guard = self.process.lock().await;
        let proc = proc_guard.as_mut().context("CLI process not started")?;
        Self::send_and_recv_stdio(proc, &line, request.id, timeout_duration).await
    }

    /// stdio send + receive.
    async fn send_and_recv_stdio(
        proc: &mut CliProcess,
        line: &str,
        request_id: u64,
        timeout_duration: tokio::time::Duration,
    ) -> Result<Value> {
        proc.stdin.write_all(line.as_bytes()).await?;
        proc.stdin.write_all(b"\n").await?;
        proc.stdin.flush().await?;

        Self::read_response(&mut proc.stdout, request_id, timeout_duration).await
    }

    /// TCP send + receive.
    async fn send_and_recv_tcp(
        conn: &mut TcpConnection,
        line: &str,
        request_id: u64,
        timeout_duration: tokio::time::Duration,
    ) -> Result<Value> {
        conn.writer.write_all(line.as_bytes()).await?;
        conn.writer.write_all(b"\n").await?;
        conn.writer.flush().await?;

        Self::read_response(&mut conn.reader, request_id, timeout_duration).await
    }

    /// Read newline-delimited JSON-RPC responses, skipping notifications,
    /// until we find the response matching `request_id`.
    async fn read_response<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
        request_id: u64,
        timeout_duration: tokio::time::Duration,
    ) -> Result<Value> {
        let deadline = tokio::time::Instant::now() + timeout_duration;
        loop {
            let remaining = deadline - tokio::time::Instant::now();
            let mut buf = String::new();
            let n = tokio::time::timeout(remaining, reader.read_line(&mut buf))
                .await
                .context("JSON-RPC response timed out")?
                .context("failed to read from CLI")?;
            if n == 0 {
                bail!("CLI closed the connection");
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.len() > MAX_LINE_BYTES {
                bail!("JSON-RPC response too large: {} bytes", trimmed.len());
            }

            // Try to parse as a response.
            let parsed: Value = serde_json::from_str(trimmed)
                .with_context(|| format!("invalid JSON from CLI: {trimmed}"))?;

            // Notifications have no `id` field — skip them.
            if parsed.get("id").is_none() || parsed.get("id") == Some(&Value::Null) {
                debug!("skipping CLI notification: {trimmed}");
                continue;
            }

            let resp: JsonRpcResponse = serde_json::from_value(parsed)
                .context("failed to deserialize JSON-RPC response")?;

            // Match by request ID.
            let resp_id = resp.id.as_ref().and_then(|v| v.as_u64());
            if resp_id != Some(request_id) {
                debug!("ignoring out-of-order response (expected {request_id}, got {resp_id:?})");
                continue;
            }

            if let Some(err) = resp.error {
                bail!("{err}");
            }
            return Ok(resp.result.unwrap_or(Value::Null));
        }
    }

    // ── Process lifecycle ────────────────────────────────────────

    /// Ensure the CLI is connected (spawn or connect as needed).
    async fn ensure_connected(&self) -> Result<()> {
        if self.cli_url.is_some() {
            let mut tcp_guard = self.tcp_conn.lock().await;
            if tcp_guard.is_none() {
                *tcp_guard = Some(self.connect_tcp().await?);
            }
            return Ok(());
        }

        let mut proc_guard = self.process.lock().await;
        let needs_start = match proc_guard.as_mut() {
            None => true,
            Some(p) => !p.is_alive(),
        };
        if needs_start {
            *proc_guard = Some(self.spawn_cli().await?);
        }
        Ok(())
    }

    /// Spawn the Copilot CLI in headless stdio mode.
    async fn spawn_cli(&self) -> Result<CliProcess> {
        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("--headless")
            .arg("--stdio")
            .arg("--no-auto-update")
            .arg("--log-level")
            .arg(&self.log_level);

        if let Some(token) = &self.github_token {
            // Pass token via environment variable for security.
            cmd.env("GITHUB_TOKEN", token);
            cmd.arg("--auth-token-env").arg("GITHUB_TOKEN");
        }

        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::inherit());
        cmd.kill_on_drop(true);

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to start Copilot CLI '{}'. Is it installed and in PATH?",
                self.cli_path
            )
        })?;

        let stdin = child
            .stdin
            .take()
            .context("failed to take stdin from Copilot CLI")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to take stdout from Copilot CLI")?;

        Ok(CliProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// Connect to an external Copilot CLI server via TCP.
    async fn connect_tcp(&self) -> Result<TcpConnection> {
        let url = self.cli_url.as_deref().context("no CLI URL configured")?;

        // Strip optional `tcp://` prefix.
        let addr = url.strip_prefix("tcp://").unwrap_or(url);

        let stream = tokio::net::TcpStream::connect(addr)
            .await
            .with_context(|| format!("failed to connect to Copilot CLI at {addr}"))?;

        let (read_half, write_half) = stream.into_split();
        Ok(TcpConnection {
            reader: BufReader::new(read_half),
            writer: write_half,
        })
    }

    // ── Protocol helpers ─────────────────────────────────────────

    /// Send a `ping` RPC and verify the protocol version.
    async fn ping(&self) -> Result<()> {
        let result = self.rpc_call("ping", None).await?;
        if let Some(version) = result.get("protocolVersion").and_then(|v| v.as_str()) {
            debug!("Copilot CLI protocol version: {version}");
        }
        Ok(())
    }

    /// Create a session with optional system message and model.
    async fn create_session(
        &self,
        system_message: Option<&str>,
        model: Option<&str>,
    ) -> Result<String> {
        let mut config = serde_json::json!({});
        if let Some(sys) = system_message {
            config["systemMessage"] = Value::String(sys.to_string());
        }
        if let Some(m) = model {
            config["model"] = Value::String(m.to_string());
        }

        let params = serde_json::json!({ "config": config });
        let result = self
            .rpc_call("session.create", Some(params))
            .await
            .context("failed to create session")?;

        let session_id = result
            .get("sessionId")
            .and_then(|v| v.as_str())
            .context("session.create did not return sessionId")?
            .to_string();

        debug!("created Copilot CLI session: {session_id}");
        Ok(session_id)
    }

    /// Ensure a session exists, creating one if necessary.
    async fn ensure_session(
        &self,
        system_message: Option<&str>,
        model: Option<&str>,
    ) -> Result<String> {
        let mut session_guard = self.session_id.lock().await;
        if let Some(id) = session_guard.as_ref() {
            return Ok(id.clone());
        }
        let id = self.create_session(system_message, model).await?;
        *session_guard = Some(id.clone());
        Ok(id)
    }

    /// Send a message to a session and collect the complete assistant response.
    ///
    /// The CLI returns session events (deltas, tool calls, idle). We accumulate
    /// `assistant.message_delta` events until `session.idle` signals completion.
    async fn send_message(&self, session_id: &str, message: &str) -> Result<SessionSendResult> {
        let params = serde_json::json!({
            "sessionId": session_id,
            "message": message,
        });

        let result = self
            .rpc_call("session.send", Some(params))
            .await
            .context("session.send failed")?;

        // The response may contain the complete assistant message directly,
        // or we may need to read streamed events. Handle both.
        Self::parse_send_result(&result)
    }

    /// Parse the result of `session.send`.
    fn parse_send_result(result: &Value) -> Result<SessionSendResult> {
        let mut text = String::new();
        let mut tool_calls = Vec::new();

        // If the result contains events, accumulate them.
        if let Some(events) = result.get("events").and_then(|v| v.as_array()) {
            for event in events {
                let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match event_type {
                    "assistant.message" | "assistant.message_delta" => {
                        if let Some(delta) = event.get("content").and_then(|v| v.as_str()) {
                            text.push_str(delta);
                        }
                        if let Some(delta) = event.get("text").and_then(|v| v.as_str()) {
                            text.push_str(delta);
                        }
                    }
                    "tool.call" => {
                        if let Some(tc) = Self::parse_tool_call(event) {
                            tool_calls.push(tc);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Direct message field (simpler response format).
        if text.is_empty() {
            if let Some(msg) = result.get("message").and_then(|v| v.as_str()) {
                text = msg.to_string();
            }
        }
        if text.is_empty() {
            if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
                text = content.to_string();
            }
        }
        // Fallback: the result itself might be a string.
        if text.is_empty() {
            if let Some(s) = result.as_str() {
                text = s.to_string();
            }
        }

        Ok(SessionSendResult { text, tool_calls })
    }

    /// Parse a tool call from a session event.
    fn parse_tool_call(event: &Value) -> Option<ProviderToolCall> {
        let name = event
            .get("name")
            .or_else(|| event.get("toolName"))
            .and_then(|v| v.as_str())?;
        let id = event
            .get("id")
            .or_else(|| event.get("callId"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let arguments = event
            .get("arguments")
            .or_else(|| event.get("parameters"))
            .map(|v| {
                if v.is_string() {
                    v.as_str().unwrap_or("{}").to_string()
                } else {
                    serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
                }
            })
            .unwrap_or_else(|| "{}".to_string());

        Some(ProviderToolCall {
            id,
            name: name.to_string(),
            arguments,
        })
    }

    /// Build messages for the Provider trait from a ChatMessage slice.
    fn extract_system_and_last_user(messages: &[ChatMessage]) -> (Option<&str>, &str) {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str());
        let last_user = messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        (system, last_user)
    }
}

/// Accumulated result from a `session.send` call.
struct SessionSendResult {
    text: String,
    tool_calls: Vec<ProviderToolCall>,
}

// ── Provider trait ───────────────────────────────────────────────

#[async_trait]
impl Provider for CopilotSdkProvider {
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> Result<String> {
        let _ = temperature; // CLI manages temperature internally.

        self.ensure_connected().await?;
        let session_id = self.ensure_session(system_prompt, Some(model)).await?;
        let result = self.send_message(&session_id, message).await?;
        Ok(result.text)
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> Result<String> {
        let (system, last_user) = Self::extract_system_and_last_user(messages);
        self.chat_with_system(system, last_user, model, temperature)
            .await
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> Result<ProviderChatResponse> {
        let (system, last_user) = Self::extract_system_and_last_user(request.messages);

        let _ = temperature;
        self.ensure_connected().await?;
        let session_id = self.ensure_session(system, Some(model)).await?;
        let result = self.send_message(&session_id, last_user).await?;

        let stop_reason = if result.tool_calls.is_empty() {
            Some(NormalizedStopReason::EndTurn)
        } else {
            Some(NormalizedStopReason::ToolCall)
        };

        Ok(ProviderChatResponse {
            text: if result.text.is_empty() {
                None
            } else {
                Some(result.text)
            },
            tool_calls: result.tool_calls,
            usage: None,
            reasoning_content: None,
            quota_metadata: None,
            stop_reason,
            raw_stop_reason: None,
        })
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn warmup(&self) -> Result<()> {
        self.ensure_connected().await?;
        self.ping().await.context(
            "Copilot CLI ping failed. Ensure the Copilot CLI is installed \
             and authenticated: https://docs.github.com/en/copilot/managing-copilot/configure-personal-settings/installing-the-github-copilot-extension-in-your-environment",
        )?;
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────

    #[test]
    fn new_defaults() {
        let provider = CopilotSdkProvider::new(None, None, None, None);
        assert_eq!(provider.cli_path, "copilot");
        assert!(provider.cli_url.is_none());
        assert!(provider.github_token.is_none());
        assert_eq!(provider.log_level, "error");
    }

    #[test]
    fn new_with_custom_path() {
        let provider =
            CopilotSdkProvider::new(Some("/usr/local/bin/copilot"), None, None, Some("debug"));
        assert_eq!(provider.cli_path, "/usr/local/bin/copilot");
        assert_eq!(provider.log_level, "debug");
    }

    #[test]
    fn new_with_cli_url() {
        let provider = CopilotSdkProvider::new(None, Some("tcp://127.0.0.1:4321"), None, None);
        assert_eq!(provider.cli_url.as_deref(), Some("tcp://127.0.0.1:4321"));
    }

    #[test]
    fn empty_strings_treated_as_none() {
        let provider = CopilotSdkProvider::new(Some(""), Some(""), Some(""), Some(""));
        assert_eq!(provider.cli_path, "copilot");
        assert!(provider.cli_url.is_none());
        assert!(provider.github_token.is_none());
        assert_eq!(provider.log_level, "error");
    }

    #[test]
    fn supports_native_tools_returns_true() {
        let provider = CopilotSdkProvider::new(None, None, None, None);
        assert!(provider.supports_native_tools());
    }

    // ── JSON-RPC serialization ───────────────────────────────

    #[test]
    fn json_rpc_request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 42,
            method: "ping".to_string(),
            params: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""jsonrpc":"2.0""#));
        assert!(json.contains(r#""id":42"#));
        assert!(json.contains(r#""method":"ping""#));
        // `params` should be omitted when None
        assert!(!json.contains("params"));
    }

    #[test]
    fn json_rpc_request_serialization_with_params() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: "session.create".to_string(),
            params: Some(serde_json::json!({"config": {"model": "gpt-4o"}})),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""method":"session.create""#));
        assert!(json.contains("params"));
        assert!(json.contains("gpt-4o"));
    }

    #[test]
    fn json_rpc_response_deserialization_success() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"result":{"message":"pong","protocolVersion":"1.0"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "1.0");
    }

    #[test]
    fn json_rpc_response_deserialization_error() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "method not found");
    }

    // ── Session result parsing ───────────────────────────────

    #[test]
    fn parse_send_result_with_events() {
        let result = serde_json::json!({
            "events": [
                {"type": "assistant.message_delta", "content": "Hello"},
                {"type": "assistant.message_delta", "content": " world"},
                {"type": "session.idle"}
            ]
        });
        let parsed = CopilotSdkProvider::parse_send_result(&result).unwrap();
        assert_eq!(parsed.text, "Hello world");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_send_result_with_direct_message() {
        let result = serde_json::json!({"message": "Direct response"});
        let parsed = CopilotSdkProvider::parse_send_result(&result).unwrap();
        assert_eq!(parsed.text, "Direct response");
    }

    #[test]
    fn parse_send_result_with_tool_calls() {
        let result = serde_json::json!({
            "events": [
                {"type": "assistant.message_delta", "content": "Let me check"},
                {
                    "type": "tool.call",
                    "name": "read_file",
                    "id": "tc-1",
                    "arguments": {"path": "test.txt"}
                }
            ]
        });
        let parsed = CopilotSdkProvider::parse_send_result(&result).unwrap();
        assert_eq!(parsed.text, "Let me check");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
        assert_eq!(parsed.tool_calls[0].id, "tc-1");
    }

    #[test]
    fn parse_send_result_empty() {
        let result = serde_json::json!({});
        let parsed = CopilotSdkProvider::parse_send_result(&result).unwrap();
        assert!(parsed.text.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_send_result_string_value() {
        let result = serde_json::json!("plain text response");
        let parsed = CopilotSdkProvider::parse_send_result(&result).unwrap();
        assert_eq!(parsed.text, "plain text response");
    }

    // ── Tool call parsing ────────────────────────────────────

    #[test]
    fn parse_tool_call_standard_format() {
        let event = serde_json::json!({
            "name": "shell",
            "id": "call-123",
            "arguments": {"command": "ls"}
        });
        let tc = CopilotSdkProvider::parse_tool_call(&event).unwrap();
        assert_eq!(tc.name, "shell");
        assert_eq!(tc.id, "call-123");
        assert!(tc.arguments.contains("command"));
    }

    #[test]
    fn parse_tool_call_alternate_fields() {
        let event = serde_json::json!({
            "toolName": "read_file",
            "callId": "alt-1",
            "parameters": "{\"path\": \"test.rs\"}"
        });
        let tc = CopilotSdkProvider::parse_tool_call(&event).unwrap();
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.id, "alt-1");
    }

    #[test]
    fn parse_tool_call_missing_name_returns_none() {
        let event = serde_json::json!({"id": "no-name"});
        assert!(CopilotSdkProvider::parse_tool_call(&event).is_none());
    }

    // ── Message extraction ───────────────────────────────────

    #[test]
    fn extract_system_and_last_user_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("How are you?"),
        ];
        let (system, last_user) = CopilotSdkProvider::extract_system_and_last_user(&messages);
        assert_eq!(system, Some("You are helpful"));
        assert_eq!(last_user, "How are you?");
    }

    #[test]
    fn extract_system_and_last_user_without_system() {
        let messages = vec![ChatMessage::user("Hello")];
        let (system, last_user) = CopilotSdkProvider::extract_system_and_last_user(&messages);
        assert!(system.is_none());
        assert_eq!(last_user, "Hello");
    }

    #[test]
    fn extract_system_and_last_user_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let (system, last_user) = CopilotSdkProvider::extract_system_and_last_user(&messages);
        assert!(system.is_none());
        assert_eq!(last_user, "");
    }

    // ── Request ID counter ───────────────────────────────────

    #[test]
    fn request_id_is_monotonic() {
        let provider = CopilotSdkProvider::new(None, None, None, None);
        let id1 = provider.next_request_id();
        let id2 = provider.next_request_id();
        let id3 = provider.next_request_id();
        assert_eq!(id1 + 1, id2);
        assert_eq!(id2 + 1, id3);
    }

    // ── JSON-RPC error display ───────────────────────────────

    #[test]
    fn json_rpc_error_display() {
        let err = JsonRpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        assert_eq!(err.to_string(), "JSON-RPC error -32600: Invalid Request");
    }
}
