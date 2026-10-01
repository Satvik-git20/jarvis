"use strict";

const DEFAULTS = { baseUrl: "http://127.0.0.1:8765", token: "", sessionId: "" };
const ASK_TIMEOUT_MS = 120_000; // a cold Ollama load can take ~25s on first ask

let cfg = { ...DEFAULTS };
let waiting = false;

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const input = $("input");
const sendBtn = $("send");
const banner = $("banner");

class ApiError extends Error {
  constructor(status, body) {
    super(typeof body === "string" && body ? body : `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

// --- config -----------------------------------------------------------------

async function loadConfig() {
  const stored = await chrome.storage.local.get(DEFAULTS);
  cfg = { ...DEFAULTS, ...stored };
  if (!cfg.sessionId) {
    cfg.sessionId = "s_ext_" + crypto.randomUUID().replaceAll("-", "").slice(0, 12);
    await chrome.storage.local.set({ sessionId: cfg.sessionId });
  }
}

// --- daemon calls -------------------------------------------------------------

async function api(path, { method = "GET", body, timeoutMs = 15_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(cfg.baseUrl + path, {
      method,
      headers: { "X-Jarvis-Token": cfg.token, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch {
    throw new ApiError(0, "daemon unreachable — start it with: uv run python -m jarvis daemon");
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    throw new ApiError(res.status, data && data.detail ? JSON.stringify(data.detail) : data);
  }
  return data;
}

async function health() {
  try {
    const res = await fetch(cfg.baseUrl + "/health", { cache: "no-store" });
    $("dot").className = "dot " + (res.ok ? "up" : "down");
  } catch {
    $("dot").className = "dot down";
  }
}

async function refreshProviders() {
  if (!cfg.token) return;
  try {
    const st = await api("/status");
    const names = (st.ready_providers || []).join(", ") || "none ready";
    $("providers").textContent = names;
  } catch {
    $("providers").textContent = "";
  }
}

// --- rendering (textContent only: model output is never parsed as HTML) -------

function bubble(role, text, meta) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const body = document.createElement("div");
  body.className = "bubble";
  body.textContent = text;
  wrap.appendChild(body);
  if (meta) {
    const line = document.createElement("div");
    line.className = "meta";
    line.textContent = meta;
    wrap.appendChild(line);
  }
  messages.appendChild(wrap);
  wrap.scrollIntoView({ block: "end" });
  return body;
}

function showHint(text) {
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = text;
  messages.appendChild(hint);
}

function clearMessages() {
  messages.replaceChildren();
}

function setBanner(text) {
  banner.textContent = text || "";
  banner.hidden = !text;
}

function friendlyError(err) {
  if (err.status === 0) return err.message;
  if (err.status === 401) return "token rejected — open Settings and paste the token from .env";
  if (err.status === 503) return "all providers exhausted — run scripts/doctor.py";
  if (err.status === 404) return "no such session";
  return `daemon error ${err.status}: ${err.message}`;
}

// --- conversation -------------------------------------------------------------

async function loadHistory() {
  clearMessages();
  if (!cfg.token) {
    setBanner("Open Settings and paste JARVIS_DAEMON_TOKEN to start chatting.");
    showHint("Click Settings, top right.");
    return;
  }
  try {
    const session = await api(`/session/${encodeURIComponent(cfg.sessionId)}`);
    if (session.messages.length === 0) {
      showHint("Ask me anything. I share history with the voice loop and opencode.");
    } else {
      for (const m of session.messages) bubble(m.role, m.content);
    }
  } catch (err) {
    if (err.status === 404) {
      showHint("Ask me anything. I share history with the voice loop and opencode.");
      return;
    }
    setBanner(friendlyError(err));
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || waiting) return;
  if (!cfg.token) {
    setBanner("Open Settings and paste JARVIS_DAEMON_TOKEN to start chatting.");
    return;
  }
  setBanner("");
  input.value = "";
  autosize();
  bubble("user", text);

  const pending = bubble("assistant", "thinking…");
  waiting = true;
  sendBtn.disabled = true;
  const started = Date.now();
  const ticker = setInterval(() => {
    pending.textContent = `thinking… ${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);

  try {
    const result = await api("/ask", {
      method: "POST",
      timeoutMs: ASK_TIMEOUT_MS,
      body: { prompt: text, session_id: cfg.sessionId },
    });
    pending.textContent = result.text;
    const wrap = pending.parentElement;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${result.provider}/${result.model} · ${result.tokens} tok · ${result.latency_ms}ms`;
    wrap.appendChild(meta);
    wrap.scrollIntoView({ block: "end" });
  } catch (err) {
    pending.textContent = friendlyError(err);
    pending.parentElement.classList.add("error");
    if (err.status === 401 || err.status === 0) health();
  } finally {
    clearInterval(ticker);
    waiting = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

async function newChat() {
  if (cfg.token) {
    try { await api(`/session/${encodeURIComponent(cfg.sessionId)}/clear`, { method: "POST" }); }
    catch { /* a brand-new session has nothing to clear */ }
  }
  clearMessages();
  showHint("New chat. History cleared for this session.");
}

// --- composer ------------------------------------------------------------------

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
}

// --- wiring --------------------------------------------------------------------

$("send").addEventListener("click", send);
$("new").addEventListener("click", newChat);
$("settings").addEventListener("click", () => chrome.runtime.openOptionsPage());
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

(async function start() {
  await loadConfig();
  await health();
  await loadHistory();
  await refreshProviders();
  setInterval(health, 30_000);
  input.focus();
})();
