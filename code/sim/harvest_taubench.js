/*
 * harvest_taubench.js -- extract per-domain sufficient statistics from real
 * tau-bench trajectories for SS15. Run in a browser console (any origin;
 * raw.githubusercontent.com allows CORS).
 *
 * Source: sierra-research/tau-bench, historical_trajectories/*.json. Each file
 * is an array of episodes {task_id, reward, info, traj, trial}; traj is a
 * message list (system/user/assistant/tool). The agent's actions are the
 * assistant turns; reward>=0.999 counts as success.
 *
 * We emit only sufficient statistics (transition counts + success-length
 * histogram) per domain -- all an absorbing-chain MLE and the first-passage KS
 * test depend on -- which keeps the transfer tiny and the fit reproducible.
 * Output is consumed by code/sim/SS15_taubench.py via data/tau_bench/stats.json.
 */
const base = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/";
const files = {
  "sonnet-35-new-retail.json": "sonnet-retail",
  "sonnet-35-new-airline.json": "sonnet-airline",
  "gpt-4o-retail.json": "gpt4o-retail",
  "gpt-4o-airline.json": "gpt4o-airline",
};
const idx = { r: 0, l: 1, w: 2, h: 3, t: 4, o: 5 };           // category -> state index
const toolName = (tc) => (tc && (tc.function ? tc.function.name : tc.name)) || "";
function cat(m) {                                              // assistant turn -> action category
  if (m.role !== "assistant") return null;
  if (m.tool_calls && m.tool_calls.length) {
    const n = toolName(m.tool_calls[0]).toLowerCase();
    if (n === "think") return "t";
    if (/transfer_to_human|transfer_to_agent/.test(n)) return "h";
    if (/(cancel|modify|return|exchange|book|update|create|delete|place|refund|reserve|send|set_)/.test(n)) return "w";
    if (/^(get|find|list|search|show|calculate|check|lookup|read)/.test(n)) return "l";
    return "o";
  }
  return "r";                                                  // plain reply to the user
}
async function harvest() {
  const domains = {};
  for (const f in files) {
    const arr = await (await fetch(base + f)).json();
    const C = Array.from({ length: 6 }, () => Array(6).fill(0));
    const Cs = Array(6).fill(0), Cf = Array(6).fill(0), pi = Array(6).fill(0);
    let n = 0, ns = 0; const slh = {};
    for (const ep of arr) {
      const seq = (ep.traj || []).map(cat).filter(Boolean).map((c) => idx[c]);
      if (seq.length < 2) continue;
      n++;
      const succ = (ep.reward !== undefined ? ep.reward : (ep.info && ep.info.reward)) >= 0.999;
      pi[seq[0]]++;
      for (let t = 0; t < seq.length - 1; t++) C[seq[t]][seq[t + 1]]++;
      if (succ) { Cs[seq[seq.length - 1]]++; ns++; slh[seq.length] = (slh[seq.length] || 0) + 1; }
      else Cf[seq[seq.length - 1]]++;
    }
    domains[files[f]] = { n, ns, C, Cs, Cf, pi, slh };
  }
  return { cats: ["r", "l", "w", "h", "t", "o"], domains };
}
// const stats = await harvest();  // then save as data/tau_bench/stats.json
