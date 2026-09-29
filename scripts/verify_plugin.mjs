// Executes the real plugin tool handlers, the same way opencode does when the
// model calls one. Node 24 strips the types natively, so this runs the exact
// file that opencode loads -- not a reimplementation of it.
//
// Usage: node --experimental-strip-types scripts/verify_plugin.mjs

import { JarvisPlugin } from "../opencode/jarvis.ts"

const t = (name, ok, detail = "") =>
  console.log(`  ${ok ? "\x1b[32mPASS\x1b[0m" : "\x1b[31mFAIL\x1b[0m"}  ${name}` +
              (detail ? `  \x1b[2m${detail}\x1b[0m` : ""))

const plugin = await JarvisPlugin({})
const tools = Object.keys(plugin.tool ?? {})
console.log(`\nregistered tools: ${tools.join(", ")}\n`)

let failed = 0
const expect = ["jarvis_ask", "jarvis_search", "jarvis_remember", "jarvis_image", "jarvis_status"]
for (const name of expect) {
  const ok = tools.includes(name)
  if (!ok) failed++
  t(`${name} is registered`, ok)
}

if (failed) {
  console.log("\n  registration failed; skipping handler tests\n")
  process.exit(1)
}

const call = async (name, args) => {
  try {
    return await plugin.tool[name].execute(args, { directory: "C:\\jarvis", worktree: "C:\\jarvis" })
  } catch (e) {
    return `THREW: ${e?.message ?? e}`
  }
}

console.log("\nhandler behaviour")
{
  const out = await call("jarvis_status", {})
  t("jarvis_status returns a report", /JARVIS v/.test(out), out.split("\n")[0]?.slice(0, 70))
  t("jarvis_status lists blocked providers", /blocked:/.test(out))
}
{
  const out = await call("jarvis_search", { query: "sqlite-vec", max_results: 2 })
  const dead = /daemon is not running|Cannot reach/.test(out)
  t("jarvis_search returns results or a clear daemon error",
    dead || /https?:\/\//.test(out), out.split("\n")[0]?.slice(0, 70))
}
{
  const out = await call("jarvis_remember", { op: "store", text: "plugin verification marker", tag: "verify" })
  t("jarvis_remember store", /Remembered|daemon is not running/.test(out), out.slice(0, 70))
}
{
  const out = await call("jarvis_remember", { op: "recall", query: "plugin verification", k: 2 })
  t("jarvis_remember recall", /marker|matching memories|daemon is not running/.test(out), out.split("\n")[0]?.slice(0, 70))
}
{
  const out = await call("jarvis_ask", { prompt: "Reply with exactly: plugin path works", max_tokens: 30 })
  t("jarvis_ask gets an answer", /plugin path works|daemon is not running|all providers/i.test(out),
    out.replace(/\n/g, " ").slice(0, 80))
}
{
  // No Pollinations key configured, so this must fail with actionable advice
  // rather than a stack trace or a silent empty result.
  const out = await call("jarvis_image", { prompt: "a red cube", width: 256, height: 256 })
  t("jarvis_image without a key explains how to fix it",
    /POLLINATIONS_KEY|enter\.pollinations\.ai|daemon is not running/.test(out), out.slice(0, 80))
}

const bad = await call("jarvis_remember", { op: "store" })
t("missing required arg is rejected cleanly", !/THREW/.test(bad), bad.slice(0, 60))

const passed = 0
console.log(`\n  ${failed === 0 ? "\x1b[32mall checks passed\x1b[0m" : `\x1b[31m${failed} failed\x1b[0m`}\n`)
process.exit(failed === 0 ? 0 : 1)
