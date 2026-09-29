// JARVIS as opencode tools.
//
// This plugin is deliberately thin. All the logic lives in the Python daemon;
// this file is a transport layer plus argument validation. Keeping it thin
// means opencode never has two copies of the same behaviour to drift.
//
// The daemon is the shared state. A jarvis_ask here and a spoken "Hey Jarvis"
// use the same session history and draw from the same provider budget.

import { tool } from "@opencode-ai/plugin"
import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"

// Where the Python side lives. Override with JARVIS_HOME if the repo moves.
const JARVIS_HOME = process.env.JARVIS_HOME ?? "C:\\jarvis"
const TIMEOUT_MS = Number(process.env.JARVIS_TIMEOUT_MS ?? 120_000)

/** Minimal .env reader -- avoids a dependency for what is a dozen lines. */
function readEnvFile(): Record<string, string> {
  const path = join(JARVIS_HOME, ".env")
  if (!existsSync(path)) return {}
  const out: Record<string, string> = {}
  for (const raw of readFileSync(path, "utf-8").split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith("#")) continue
    const eq = line.indexOf("=")
    if (eq < 1) continue
    const key = line.slice(0, eq).trim()
    let val = line.slice(eq + 1).trim()
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    )
      val = val.slice(1, -1)
    out[key] = val
  }
  return out
}

function config() {
  const file = readEnvFile()
  const pick = (k: string, fallback = "") => process.env[k] || file[k] || fallback
  return {
    base: `http://${pick("JARVIS_DAEMON_HOST", "127.0.0.1")}:${pick("JARVIS_DAEMON_PORT", "8765")}`,
    token: pick("JARVIS_DAEMON_TOKEN"),
  }
}

const DAEMON_DOWN = [
  "JARVIS daemon is not running, so this tool cannot work.",
  "",
  "Start it with:",
  "  cd " + JARVIS_HOME + " && uv run python -m jarvis.daemon",
  "",
  "Then check it with:",
  "  uv run python " + JARVIS_HOME + "\\scripts\\doctor.py",
].join("\n")

const NO_TOKEN =
  "JARVIS_DAEMON_TOKEN is not set, so the daemon rejected this request. " +
  "Add a token to " + join(JARVIS_HOME, ".env") + " and restart the daemon."

/** One HTTP call, with timeouts and errors turned into readable text. */
async function call(
  path: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<{ ok: boolean; status: number; data: any }> {
  const { base, token } = config()
  if (!token) return { ok: false, status: 0, data: { detail: NO_TOKEN } }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), init.timeoutMs ?? TIMEOUT_MS)
  try {
    const res = await fetch(base + path, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "X-Jarvis-Token": token,
        ...(init.headers ?? {}),
      },
    })
    const text = await res.text()
    let data: any
    try {
      data = text ? JSON.parse(text) : {}
    } catch {
      data = { detail: text.slice(0, 400) }
    }
    return { ok: res.ok, status: res.status, data }
  } catch (err: any) {
    const aborted = err?.name === "AbortError"
    return {
      ok: false,
      status: 0,
      data: {
        detail: aborted
          ? `JARVIS did not respond within ${(init.timeoutMs ?? TIMEOUT_MS) / 1000}s.`
          : `Cannot reach the JARVIS daemon at ${base}.\n\n${DAEMON_DOWN}`,
        unreachable: true,
      },
    }
  } finally {
    clearTimeout(timer)
  }
}

function renderError(path: string, res: { status: number; data: any }): string {
  const d = res.data?.detail
  if (typeof d === "string") return d
  if (d?.error === "all providers exhausted") {
    const attempts = (d.attempts ?? []).map(([n, why]: [string, string]) => `  - ${n}: ${why}`)
    return [
      "JARVIS could not get an answer: every free provider is unavailable.",
      "",
      ...attempts,
      "",
      d.hint ?? "",
    ].join("\n")
  }
  return `JARVIS ${path} failed (HTTP ${res.status}): ${JSON.stringify(d).slice(0, 400)}`
}

export const JarvisPlugin = async () => {
  return {
    tool: {
      // --- the brain ----------------------------------------------------
      jarvis_ask: tool({
        description:
          "Ask JARVIS, a separate voice assistant that runs on free AI providers. " +
          "Use this for open-ended questions, factual lookups, summarising, or " +
          "when you want a second opinion outside your own context window. " +
          "It routes across local Ollama and free cloud models and fails over " +
          "automatically. For simple lookups prefer your own tools; use this " +
          "when a second perspective or a long-form answer is useful.",
        args: {
          prompt: tool.schema.string().describe("The question or instruction for JARVIS."),
          session_id: tool.schema
            .string()
            .optional()
            .describe(
              "Continue an existing JARVIS conversation for context. Omit to start fresh.",
            ),
          prefer: tool.schema
            .enum(["ollama", "openrouter", "groq", "cerebras", "gemini"])
            .optional()
            .describe("Force a specific provider. Omit to let the router choose."),
          max_tokens: tool.schema
            .number()
            .optional()
            .describe("Response length cap. Default 1024."),
        },
        async execute(args) {
          const res = await call("/ask", {
            method: "POST",
            body: JSON.stringify({
              prompt: args.prompt,
              session_id: args.session_id,
              prefer: args.prefer,
              max_tokens: args.max_tokens,
            }),
          })
          if (!res.ok) return renderError("/ask", res)
          const d = res.data
          return `${d.text}\n\n---\nvia ${d.provider}/${d.model} · ${d.tokens} tokens · ${d.latency_ms}ms · session ${d.session_id}`
        },
      }),

      // --- live web ----------------------------------------------------
      jarvis_search: tool({
        description:
          "Search the live web through JARVIS (Tavily, then Jina, then DuckDuckGo). " +
          "Returns titles, URLs and snippets. Use when the answer depends on " +
          "current information, or when you need a source to cite. Set " +
          "fetch_content=true to pull readable page text for the top results.",
        args: {
          query: tool.schema.string().describe("What to search for."),
          max_results: tool.schema.number().optional().describe("Default 5, max 10."),
          fetch_content: tool.schema
            .boolean()
            .optional()
            .describe("Also fetch full page text for the top results. Slower."),
        },
        async execute(args) {
          const res = await call("/search", {
            method: "POST",
            body: JSON.stringify({
              query: args.query,
              max_results: args.max_results ?? 5,
              fetch_content: args.fetch_content ?? false,
            }),
            timeoutMs: args.fetch_content ? 180_000 : 60_000,
          })
          if (!res.ok) return renderError("/search", res)
          if (!res.data.count) return `No results for "${args.query}".`
          return res.data.results
            .map(
              (r: any, i: number) =>
                `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}` +
                (r.content ? `\n   [content] ${r.content.slice(0, 800)}` : ""),
            )
            .join("\n\n")
        },
      }),

      // --- long-term memory --------------------------------------------
      jarvis_remember: tool({
        description:
          "JARVIS's long-term memory, shared with its voice interface. " +
          "op=store saves a durable fact worth recalling later (user preferences, " +
          "project constraints, decisions and their reasons). " +
          "op=recall does semantic search over everything stored. " +
          "op=forget deletes matching memories. " +
          "Use store for facts that should outlive this conversation.",
        args: {
          op: tool.schema
            .enum(["store", "recall", "forget"])
            .describe("store, recall, or forget."),
          text: tool.schema.string().optional().describe("Required for op=store."),
          query: tool.schema
            .string()
            .optional()
            .describe("Required for op=recall and op=forget."),
          tag: tool.schema.string().optional().describe("Optional label, e.g. 'prefs'."),
          k: tool.schema.number().optional().describe("Results to return. Default 5."),
        },
        async execute(args) {
          const res = await call("/remember", {
            method: "POST",
            body: JSON.stringify({
              op: args.op,
              text: args.text,
              query: args.query,
              tag: args.tag,
              k: args.k ?? 5,
            }),
          })
          if (!res.ok) return renderError("/remember", res)
          const d = res.data
          if (d.op === "store")
            return d.embedded
              ? `Remembered (${d.id}) via ${d.backend}.`
              : `Remembered (${d.id}) but ${d.note}`
          if (d.op === "forget") return `Removed ${d.removed} memory/memories.`
          if (!d.count) return "No matching memories."
          return d.memories
            .map(
              (m: any) =>
                `- [${m.score ?? "kw"}] ${m.text}${m.tag ? `  (${m.tag})` : ""}`,
            )
            .join("\n")
        },
      }),

      // --- images -------------------------------------------------------
      jarvis_image: tool({
        description:
          "Generate an image with JARVIS via Pollinations (free flux model) and " +
          "save it to disk. Use when the user asks for an image, illustration, " +
          "diagram or mockup. Returns a local file path, not raw image data.",
        args: {
          prompt: tool.schema.string().describe("What the image should show."),
          width: tool.schema.number().optional().describe("Default 1024."),
          height: tool.schema.number().optional().describe("Default 1024."),
        },
        async execute(args) {
          const res = await call("/image", {
            method: "POST",
            body: JSON.stringify({
              prompt: args.prompt,
              width: args.width ?? 1024,
              height: args.height ?? 1024,
            }),
            timeoutMs: 180_000,
          })
          if (!res.ok) return renderError("/image", res)
          const d = res.data
          return `Saved to ${d.path} (${d.width}x${d.height}, ${Math.round(d.bytes / 1024)} KB, ${d.model})`
        },
      }),

      // --- diagnostics --------------------------------------------------
      jarvis_status: tool({
        description:
          "Report JARVIS's health: which free AI providers are ready, which are " +
          "blocked and why, remaining daily rate-limit budget per provider, and " +
          "any outstanding setup steps. Use this to debug why a JARVIS tool " +
          "failed or to check whether a provider still has quota.",
        args: {},
        async execute() {
          const res = await call("/status", { timeoutMs: 30_000 })
          if (!res.ok) return renderError("/status", res)
          const d = res.data
          const lines = [
            `JARVIS v${d.version} · up ${Math.round(d.uptime_seconds)}s`,
            `ready:   ${(d.ready_providers ?? []).join(", ") || "none"}`,
            `blocked: ${
              Object.entries(d.blocked_providers ?? {})
                .map(([k, v]) => `${k} (${v})`)
                .join(", ") || "none"
            }`,
          ]
          const budgets = Object.entries(d.budget ?? {})
            .filter(([, v]: any) => v.total_requests || v.errors)
            .map(
              ([k, v]: any) =>
                `  ${k}: ${v.total_requests} calls, ${v.total_tokens} tokens, ` +
                `${v.errors} errors${v.cooldown_seconds > 0 ? `, ${v.cooldown_seconds}s cooldown` : ""}` +
                `${v.last_error ? `\n     last error: ${v.last_error.slice(0, 160)}` : ""}`,
            )
          if (budgets.length) lines.push("usage:", ...budgets)
          if (d.setup_hints?.length) lines.push("setup needed:", ...d.setup_hints.map((h: string) => `  - ${h}`))
          return lines.join("\n")
        },
      }),
    },
  }
}
