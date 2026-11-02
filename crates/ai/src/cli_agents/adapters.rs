//! How each agent CLI is driven: its arguments, the flags it must have, how its output stream
//! reads, and anything it needs before a run (a sign-in check). Each is an [`AgentAdapter`]; an
//! [`AgentAdapters`] registry maps agent ids to them, and an agent it doesn't know gets the plain
//! adapter (prompt in, text out). Add an agent by implementing the trait and registering it.

use super::*;

mod antigravity;
mod hermes;
mod opencode;
pub use antigravity::Antigravity;
pub use hermes::Hermes;
pub use opencode::OpenCode;

/// What a run's arguments are built from.
pub struct ArgsCx<'a> {
    pub options: &'a BackendOptions,
    /// The message, already protected from being read as a flag.
    pub prompt: &'a str,
    pub preamble: &'a str,
    pub mcp: &'a McpCommand,
    /// A saved session to continue (only passed to adapters that [`resume`](AgentAdapter::resumes)).
    pub resume: Option<&'a str>,
    /// Build mode: the agent may edit files; approvals and questions come to us.
    pub builder: bool,
    /// Clean room: reads go through our approval tool too, so the original stays out of reach.
    pub reads_checked: bool,
    /// A question about a build: only reading (none in Clean room), a resumed session forked and
    /// not kept, no tool server.
    pub question: bool,
}

/// One way of driving an agent CLI.
pub trait AgentAdapter: Send + Sync {
    /// The agent id this adapter is registered under (`claude-code`), plus any other names.
    fn ids(&self) -> &'static [&'static str];
    /// argv after the program for one message.
    fn args(&self, cx: &ArgsCx) -> Vec<String>;
    /// `(help invocation, substrings it must mention)`, checked once before the first run; when
    /// they're missing the run falls back to the plain adapter. Empty: nothing to check.
    fn required_flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        (&[], &[])
    }
    /// `program <prompt>` answers and exits, so the plain adapter is a fallback. When it opens an
    /// interactive session instead, a failed flag check keeps this adapter, and the run says what
    /// went wrong (#199).
    fn plain_fallback(&self) -> bool {
        true
    }
    /// The CLI writes one JSON event per line (else its output is the answer text).
    fn streams_json(&self) -> bool {
        true
    }
    /// Events for one parsed JSON line.
    fn parse(&self, _p: &mut StreamParser, _v: &Value) -> Vec<AgentEvent> {
        Vec::new()
    }
    /// It can continue a saved session.
    fn resumes(&self) -> bool {
        false
    }
    /// It honours [`ArgsCx::question`]: it can be held to reading only, and a resumed session
    /// isn't changed. Questions about a build go only to adapters that say so.
    fn answers_questions(&self) -> bool {
        false
    }
    /// The process for a local run of `program` (an adapter may clean its environment).
    fn process(&self, program: &str) -> Process {
        Process::new(program)
    }
    /// Whether it's signed in, checked before a build starts. `None`: can't tell (the run itself
    /// reports any sign-in problem).
    fn signed_in(&self, _program: &str) -> Option<bool> {
        None
    }
    /// The plain-words fix when [`signed_in`](AgentAdapter::signed_in) says no.
    fn sign_in_fix(&self) -> Option<String> {
        None
    }
    /// Whether what a build needs from this computer is there, for an agent that's handed our tool
    /// server, checked before a build starts. An agent that isn't handed it has nothing to check.
    /// `stop` ends the check when the person stops the build.
    fn check_tool_server(&self, _stop: &CancelToken) -> Result<(), ToolError> {
        Ok(())
    }
}

/// The adapters a run can use.
#[derive(Clone)]
pub struct AgentAdapters {
    adapters: Vec<Arc<dyn AgentAdapter>>,
}

impl std::fmt::Debug for AgentAdapters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.adapters.iter().map(|a| a.ids().first().copied().unwrap_or_default())).finish()
    }
}

impl Default for AgentAdapters {
    fn default() -> Self {
        AgentAdapters::built_in()
    }
}

impl AgentAdapters {
    /// Claude Code, Codex, Gemini CLI, OpenCode, Antigravity and Hermes.
    pub fn built_in() -> AgentAdapters {
        AgentAdapters {
            adapters: vec![Arc::new(ClaudeCode), Arc::new(Codex), Arc::new(GeminiCli), Arc::new(OpenCode), Arc::new(Antigravity), Arc::new(Hermes)],
        }
    }

    /// Add an adapter; it takes over any ids it shares with one already registered.
    pub fn register(&mut self, adapter: Arc<dyn AgentAdapter>) {
        self.adapters.retain(|a| !a.ids().iter().any(|id| adapter.ids().contains(id)));
        self.adapters.push(adapter);
    }

    /// The adapter for an agent id; one it doesn't know gets the plain adapter.
    pub fn for_agent(&self, agent: &str) -> Arc<dyn AgentAdapter> {
        self.adapters.iter().find(|a| a.ids().contains(&agent)).cloned().unwrap_or_else(|| Arc::new(Plain))
    }
}

/// Any CLI: the prompt as its last argument, its output streamed as text.
pub struct Plain;

impl AgentAdapter for Plain {
    fn ids(&self) -> &'static [&'static str] {
        &[]
    }
    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut a = cx.options.args.clone();
        a.push(format!("{}{}", cx.preamble, cx.prompt));
        a
    }
    fn streams_json(&self) -> bool {
        false
    }
}

/// Claude Code settings with one PostToolUse hook, run after every tool call: our CLI hands over
/// the notes waiting in `dir` (`craft-anything-cli builder-notes <dir>`). Paths use forward
/// slashes and quotes so the hook's shell reads them the same on every platform.
pub fn notes_hook_settings(cli: &str, dir: &std::path::Path) -> String {
    let fwd = |s: &str| s.replace('\\', "/");
    let command = format!("\"{}\" builder-notes \"{}\"", fwd(cli), fwd(&dir.display().to_string()));
    json!({"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": 10}]}]}}).to_string()
}

/// Claude Code's tools that change something or reach outside, refused for a question.
const CHANGING_TOOLS: &[&str] =
    &["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "PowerShell", "WebFetch", "WebSearch", "Task", "TodoWrite", "AskUserQuestion"];

/// Claude Code answering a question about a build: the builder's session copied
/// (`--fork-session`) and the copy not kept, no tool server, and only Read, Glob and Grep (none in
/// Clean room).
fn question_args(mut a: Vec<String>, cx: &ArgsCx) -> Vec<String> {
    a.extend([
        "-p".into(),
        cx.prompt.to_string(),
        "--output-format".into(),
        "stream-json".into(),
        "--verbose".into(),
        "--append-system-prompt".into(),
        cx.preamble.to_string(),
        "--model".into(),
        cx.options.cli_model.clone().unwrap_or_else(|| crate::anthropic::DEFAULT_MODEL.into()),
    ]);
    if let Some(r) = cx.resume {
        a.extend(["--resume".into(), r.to_string(), "--fork-session".into()]);
    }
    a.extend(["--no-session-persistence".into(), "--strict-mcp-config".into(), "--disallowedTools".into()]);
    a.extend(CHANGING_TOOLS.iter().map(|t| t.to_string()));
    if !cx.reads_checked {
        a.extend(["--allowedTools", "Read", "Glob", "Grep"].map(String::from));
    }
    a
}

/// Claude Code in headless mode (flags checked against `claude --help`).
pub struct ClaudeCode;

impl AgentAdapter for ClaudeCode {
    fn ids(&self) -> &'static [&'static str] {
        &["claude-code", "claude"]
    }

    fn answers_questions(&self) -> bool {
        true
    }

    fn check_tool_server(&self, stop: &CancelToken) -> Result<(), ToolError> {
        cli_can_start(craft_anything_platform::self_cli_path().as_deref(), stop)
    }

    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut a = cx.options.args.clone();
        if cx.question {
            return question_args(a, cx);
        }
        let mcp_json = json!({"mcpServers": {MCP_NAME: {"command": cx.mcp.command, "args": cx.mcp.args}}}).to_string();
        a.extend([
            "-p".into(),
            cx.prompt.to_string(),
            "--output-format".into(),
            "stream-json".into(),
            "--verbose".into(),
            "--append-system-prompt".into(),
            cx.preamble.to_string(),
        ]);
        // Opus 5.5 everywhere unless config.toml names another model on purpose; the effort
        // goes in through the environment (see `run_cli`).
        a.extend(["--model".into(), cx.options.cli_model.clone().unwrap_or_else(|| crate::anthropic::DEFAULT_MODEL.into())]);
        if let Some(r) = cx.resume {
            a.extend(["--resume".into(), r.to_string()]);
        }
        if cx.builder {
            // Questions go through our core_person_ask tool, which the app shows as buttons.
            a.extend(["--permission-prompt-tool".into(), format!("mcp__{MCP_NAME}__{APPROVAL_TOOL}"), "--disallowedTools".into(), "AskUserQuestion".into()]);
        }
        if let Some(dir) = cx.mcp.notes.as_deref().filter(|_| cx.builder) {
            // After every tool call, notes the person added meanwhile reach the model as context.
            a.extend(["--settings".into(), notes_hook_settings(&cx.mcp.command, dir)]);
        }
        a.extend(["--strict-mcp-config".into(), "--mcp-config".into(), mcp_json, "--allowedTools".into(), format!("mcp__{MCP_NAME}__*")]);
        if cx.builder && cx.reads_checked {
            a.push("TodoWrite".into());
        } else if cx.builder {
            a.extend(["Read", "Glob", "Grep", "TodoWrite"].map(String::from));
        }
        a
    }

    fn parse(&self, p: &mut StreamParser, v: &Value) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        if let Some(id) = v.get("session_id").and_then(Value::as_str) {
            p.session_id = Some(id.to_string());
        }
        match s("type").as_str() {
            "assistant" => {
                for b in v.pointer("/message/content").and_then(Value::as_array).cloned().unwrap_or_default() {
                    match b.get("type").and_then(Value::as_str).unwrap_or_default() {
                        "text" => out.push(AgentEvent::TextDelta { text: b.get("text").and_then(Value::as_str).unwrap_or_default().to_string() }),
                        "tool_use" => {
                            let id = b.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                            let name = b.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                            out.push(p.started(&id, &name, b.get("input").cloned().unwrap_or(Value::Null)));
                        }
                        _ => {}
                    }
                }
            }
            "user" => {
                for b in v.pointer("/message/content").and_then(Value::as_array).cloned().unwrap_or_default() {
                    if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                        let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or_default().to_string();
                        let err = b.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                        let text = content_text(b.get("content").unwrap_or(&Value::Null));
                        out.push(p.finished(&id, !err, &text));
                    }
                }
            }
            "result" => {
                p.saw_result = true;
                p.cost_usd = v.get("total_cost_usd").and_then(Value::as_f64);
                if let Some(u) = v.get("usage") {
                    p.input_tokens = u
                        .get("input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        .saturating_add(u.get("cache_read_input_tokens").and_then(Value::as_u64).unwrap_or(0));
                    p.output_tokens = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                }
                if v.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
                    let why = if s("result").is_empty() { s("subtype") } else { s("result") };
                    let (message, remediation) = claude_code_failure(&why);
                    out.push(AgentEvent::Error { message, remediation, log: None });
                }
            }
            _ => {}
        }
        out
    }

    fn resumes(&self) -> bool {
        true
    }

    /// The person's own standalone Claude Code (see [`host_session_vars`]).
    fn process(&self, program: &str) -> Process {
        standalone(program)
    }

    fn signed_in(&self, program: &str) -> Option<bool> {
        let out = standalone(program).args(["auth", "status", "--json"]).timeout(Duration::from_secs(15)).stdout_cap(64 * 1024).run().ok()?;
        let v: Value = serde_json::from_str(out.stdout_text().trim()).ok()?;
        v.get("loggedIn").and_then(Value::as_bool)
    }

    fn sign_in_fix(&self) -> Option<String> {
        sign_in_hint("login")
    }
}

/// Codex (`codex exec --json`). Flags are checked against `codex exec --help` before use.
pub struct Codex;

impl AgentAdapter for Codex {
    fn ids(&self) -> &'static [&'static str] {
        &["codex"]
    }

    fn check_tool_server(&self, stop: &CancelToken) -> Result<(), ToolError> {
        cli_can_start(craft_anything_platform::self_cli_path().as_deref(), stop)
    }

    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut v = vec!["exec".to_string(), "--json".into()];
        // Project folders aren't git repositories, and `codex exec` refuses to run outside one
        // without this (#69). Not twice: it may already be in the backend's own args.
        const SKIP_GIT: &str = "--skip-git-repo-check";
        if !cx.options.args.iter().any(|a| a == SKIP_GIT) {
            v.push(SKIP_GIT.into());
        }
        if cx.builder {
            v.extend(["--sandbox".into(), "workspace-write".into()]);
        }
        v.extend(["-c".into(), format!("mcp_servers.{MCP_NAME}.command={}", toml_string(&cx.mcp.command))]);
        let args_toml = format!("[{}]", cx.mcp.args.iter().map(|s| toml_string(s)).collect::<Vec<_>>().join(", "));
        v.extend(["-c".into(), format!("mcp_servers.{MCP_NAME}.args={args_toml}")]);
        if let Some(m) = &cx.options.cli_model {
            v.extend(["-m".into(), m.clone()]);
        }
        v.extend(cx.options.args.iter().cloned());
        v.push(format!("{}{}", cx.preamble, cx.prompt));
        v
    }

    fn required_flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        (&["exec", "--help"], &["--json"])
    }

    /// Plain `codex <prompt>` is the interactive TUI: "stdin is not a terminal".
    fn plain_fallback(&self) -> bool {
        false
    }

    fn parse(&self, p: &mut StreamParser, v: &Value) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let t = v.get("type").and_then(Value::as_str).unwrap_or_default();
        if let Some(id) = v.get("thread_id").and_then(Value::as_str) {
            p.session_id = Some(id.to_string());
        }
        let item = v.get("item").cloned().unwrap_or(Value::Null);
        let it = item.get("type").and_then(Value::as_str).unwrap_or_default();
        let id = item.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
        match (t, it) {
            ("item.completed", "agent_message") => {
                out.push(AgentEvent::TextDelta { text: format!("{}\n", item.get("text").and_then(Value::as_str).unwrap_or_default()) })
            }
            ("item.completed", "reasoning") => out.push(AgentEvent::Status { text: summary_of(item.get("text").and_then(Value::as_str).unwrap_or_default()) }),
            ("item.started", "mcp_tool_call") => {
                let name = item.get("tool").and_then(Value::as_str).unwrap_or_default().to_string();
                out.push(p.started(&id, &name, item.get("arguments").cloned().unwrap_or(Value::Null)));
            }
            ("item.completed", "mcp_tool_call") => {
                if !p.tool_names.contains_key(&id) {
                    let name = item.get("tool").and_then(Value::as_str).unwrap_or_default().to_string();
                    out.push(p.started(&id, &name, item.get("arguments").cloned().unwrap_or(Value::Null)));
                }
                let failed = item.get("status").and_then(Value::as_str) == Some("failed") || item.get("error").is_some_and(|e| !e.is_null());
                let text = item.get("result").or_else(|| item.get("error")).map(|r| r.to_string()).unwrap_or_default();
                out.push(p.finished(&id, !failed, &text));
            }
            ("item.started", "command_execution") => {
                out.push(AgentEvent::Status { text: format!("Running: {}", summary_of(item.get("command").and_then(Value::as_str).unwrap_or_default())) })
            }
            ("turn.completed", _) => {
                p.saw_result = true;
                if let Some(u) = v.get("usage") {
                    p.input_tokens = p.input_tokens.saturating_add(u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0));
                    p.output_tokens = p.output_tokens.saturating_add(u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0));
                }
            }
            ("turn.failed", _) | ("error", _) => {
                let m = v.pointer("/error/message").or_else(|| v.get("message")).and_then(Value::as_str).unwrap_or("unknown error");
                out.push(AgentEvent::Error { message: format!("Codex: {m}"), remediation: None, log: None });
            }
            _ => {}
        }
        out
    }
}

/// Gemini CLI (`--output-format stream-json`). Flags are checked against `gemini --help`.
pub struct GeminiCli;

impl AgentAdapter for GeminiCli {
    fn ids(&self) -> &'static [&'static str] {
        &["gemini-cli", "gemini"]
    }

    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut a = cx.options.args.clone();
        if let Some(m) = &cx.options.cli_model {
            a.extend(["-m".into(), m.clone()]);
        }
        a.extend(["--output-format".into(), "stream-json".into(), "-p".into(), format!("{}{}", cx.preamble, cx.prompt)]);
        a
    }

    fn required_flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        (&["--help"], &["--output-format", "stream-json"])
    }

    fn parse(&self, p: &mut StreamParser, v: &Value) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        match s("type").as_str() {
            "init" => p.session_id = v.get("session_id").and_then(Value::as_str).map(str::to_string),
            "message" if s("role") == "assistant" => out.push(AgentEvent::TextDelta { text: s("content") }),
            "tool_use" => {
                let id = s("tool_id");
                let name = s("tool_name");
                out.push(p.started(&id, &name, v.get("parameters").cloned().unwrap_or(Value::Null)));
            }
            "tool_result" => {
                let id = s("tool_id");
                let ok = s("status") != "error";
                let text = v.get("output").map(content_text).unwrap_or_default();
                out.push(p.finished(&id, ok, &text));
            }
            "error" => out.push(AgentEvent::Error { message: format!("Gemini CLI: {}", s("message")), remediation: None, log: None }),
            "result" => {
                p.saw_result = true;
                if let Some(st) = v.get("stats") {
                    p.input_tokens = st.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                    p.output_tokens = st.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                }
            }
            _ => {}
        }
        out
    }
}
