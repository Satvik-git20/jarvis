# JARVIS Chrome side panel

A chat panel for the JARVIS daemon. Same sessions, same provider budget, same
history as the voice loop and opencode — it is another client of
`http://127.0.0.1:8765`, not a second brain.

## Install (unpacked)

1. Start the daemon: `uv run python -m jarvis daemon`
2. Open `chrome://extensions`, turn on **Developer mode**
3. **Load unpacked** → select this `chrome-extension/` folder
4. Click the JARVIS toolbar icon (the panel opens beside your tabs)
5. **Settings** → paste the value of `JARVIS_DAEMON_TOKEN` from `.env` →
   **Test connection**

## How it talks to the daemon

- Every request sends `X-Jarvis-Token`; the token is stored in
  `chrome.storage.local` for this browser profile only and is never synced.
- `host_permissions` covers `http://127.0.0.1:8765/*`, and the daemon's CORS
  allows `chrome-extension://` origins only — ordinary web pages can reach the
  loopback port but cannot read a single byte back.
- The panel keeps one session id (`s_ext_…`), so "New chat" clears that
  session server-side and the next prompt continues in the same shared store
  the voice loop sees.

## Files

| File | Role |
|---|---|
| `manifest.json` | MV3 manifest: side panel, storage, localhost access |
| `background.js` | service worker: toolbar click opens the panel |
| `sidepanel.html/css/js` | the chat panel |
| `options.html/js` | daemon URL + token settings |
