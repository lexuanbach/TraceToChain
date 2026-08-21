/*
 * harvest_swebench.js — harvest real SWE-agent trajectories for SS10.
 *
 * Run in a browser DevTools console on a page whose origin is the public
 * SWE-bench submissions bucket, e.g. after navigating to:
 *   https://swe-bench-submissions.s3.amazonaws.com/?list-type=2&prefix=verified/&delimiter=/
 * (the bucket is public / anonymous-readable, so same-origin fetch works).
 *
 * It lists every instance under the chosen submission's trajs/ folder,
 * fetches each <id>.traj, and extracts ONE canonical action token per step
 * (no observations are kept). Ground-truth resolved labels are read from the
 * submission's results.json on raw.githubusercontent.com (CORS-enabled).
 *
 * Output: a plain-text payload (LEGEND line + "<id> <S|F> <seqchars>" lines)
 * suitable for code/sim/_build_corpus.py.
 */
const BASE = "https://swe-bench-submissions.s3.amazonaws.com/";
const SUB  = "verified/20250522_sweagent_claude-4-sonnet-20250514/";
const RESULTS =
  "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/" +
  SUB + "results/results.json";

// canonical action token for one step action string
function metaTok(a) {
  if (a == null) return "none";
  let s = ("" + a).trim(); if (!s) return "none";
  let line = s.split("\n").map(x => x.trim()).find(x => x.length) || "";
  let prev; do { prev = line; line = line.replace(/^cd\s+[^&|;]+&&\s*/, ""); } while (line !== prev);
  line = line.replace(/^sudo\s+/, "").replace(/^source\s+\S+\s*&&\s*/, "");
  let toks = line.split(/\s+/), cmd = (toks[0] || "").replace(/^\.?\/?/, "");
  if (cmd === "str_replace_editor" || cmd === "str_replace_based_edit_tool" ||
      cmd === "edit" || cmd === "insert") {
    let sc = (toks[1] || "").replace(/[^a-zA-Z_]/g, ""); return "edit:" + (sc || "?");
  }
  return cmd.slice(0, 24);
}

async function listInstances() {
  let out = [], token = null;
  do {
    let u = BASE + "?list-type=2&delimiter=/&prefix=" + encodeURIComponent(SUB + "trajs/");
    if (token) u += "&continuation-token=" + encodeURIComponent(token);
    const d = new DOMParser().parseFromString(await (await fetch(u)).text(), "application/xml");
    d.querySelectorAll("CommonPrefixes > Prefix").forEach(p => {
      const parts = p.textContent.split("/"); out.push(parts[parts.length - 2]);
    });
    const t = d.querySelector("NextContinuationToken"); token = t ? t.textContent : null;
  } while (token);
  return out;
}

async function harvest(concurrency = 10) {
  const resolved = new Set((await (await fetch(RESULTS)).json()).resolved || []);
  const insts = await listInstances();
  const data = {};
  let i = 0;
  const worker = async () => {
    while (i < insts.length) {
      const inst = insts[i++];
      try {
        const j = await (await fetch(BASE + SUB + "trajs/" + inst + "/" + inst + ".traj")).json();
        data[inst] = { toks: (j.trajectory || []).map(s => metaTok(s.action)),
                       resolved: resolved.has(inst) };
      } catch (e) { /* skip */ }
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  // serialize as LEGEND + lines
  const legend = {}; let n = 0; const lines = [];
  for (const inst of Object.keys(data).sort()) {
    const seq = data[inst].toks.map(t => { if (!(t in legend)) legend[t] = n++; return String.fromCharCode(97 + legend[t]); }).join("");
    lines.push(inst + " " + (data[inst].resolved ? "S" : "F") + " " + seq);
  }
  const inv = {}; for (const k in legend) inv[legend[k]] = k;
  return "LEGEND " + JSON.stringify(inv) + "\n" + lines.join("\n");
}

// const payload = await harvest(); // then copy/save payload
