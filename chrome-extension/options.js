"use strict";

const DEFAULTS = { baseUrl: "http://127.0.0.1:8765", token: "" };
const $ = (id) => document.getElementById(id);

function result(text, ok) {
  const el = $("result");
  el.textContent = text;
  el.className = ok === true ? "ok" : ok === false ? "bad" : "";
}

async function save() {
  const baseUrl = $("url").value.trim().replace(/\/+$/, "") || DEFAULTS.baseUrl;
  const token = $("token").value.trim();
  await chrome.storage.local.set({ baseUrl, token });
  result("saved", true);
}

async function test() {
  const baseUrl = $("url").value.trim().replace(/\/+$/, "") || DEFAULTS.baseUrl;
  const token = $("token").value.trim();
  result("testing…");
  let res;
  try {
    res = await fetch(baseUrl + "/status", { headers: { "X-Jarvis-Token": token } });
  } catch {
    result("unreachable — is the daemon running?", false);
    return;
  }
  if (res.status === 401) {
    result("token rejected", false);
    return;
  }
  if (!res.ok) {
    result(`HTTP ${res.status}`, false);
    return;
  }
  const st = await res.json();
  result(`ok — ${st.sessions} session(s), providers: ${(st.ready_providers || []).join(", ") || "none"}`, true);
}

(async function start() {
  const stored = await chrome.storage.local.get(DEFAULTS);
  $("url").value = stored.baseUrl || DEFAULTS.baseUrl;
  $("token").value = stored.token || "";
})();

$("save").addEventListener("click", save);
$("test").addEventListener("click", test);
