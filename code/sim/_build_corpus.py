"""Reassemble the transferred SWE-bench corpus chunks into corpus.json.

The browser-side harvester serialized the corpus as a plain-text payload
(LEGEND line + one "<instance> <S|F> <seqchars>" line per trace) and emitted
it in ~900-char chunks marked PXC<nnn>~ ... ~END<nnn>. This script:
  1. concatenates the chunk payloads in index order (rejoining boundary splits),
  2. verifies length + checksum against the values reported by the browser,
  3. decodes each sequence char (chr(97+token_id)) back to its action token
     via the embedded LEGEND, and
  4. writes data/swebench_real/corpus.json = {inst: {toks, resolved, label}}.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "swebench_real"
EXPECT_LEN = 41941
EXPECT_CKSUM = 3732821080

raw = ""
for f in ["chunks.txt", "chunks2.txt"]:
    raw += (DATA / f).read_text()

# extract PXC<nnn>~payload~END<nnn>
chunks = {}
for m in re.finditer(r"PXC(\d{3})~(.*?)~END\1", raw, re.DOTALL):
    chunks[int(m.group(1))] = m.group(2)
idx = sorted(chunks)
assert idx == list(range(len(idx))), f"missing/extra chunks: have {len(idx)}, max {idx[-1]}"
payload = "".join(chunks[i] for i in idx)

def cksum(s):
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h

got_len, got_ck = len(payload), cksum(payload)
print(f"reassembled: len={got_len} (expect {EXPECT_LEN})  checksum={got_ck} (expect {EXPECT_CKSUM})")
print(f"  length_match={got_len==EXPECT_LEN}  checksum_match={got_ck==EXPECT_CKSUM}")

lines = payload.split("\n")
assert lines[0].startswith("LEGEND "), "missing LEGEND header"
legend = json.loads(lines[0][len("LEGEND "):])
id2tok = {int(k): v for k, v in legend.items()}
char2tok = {chr(97 + i): id2tok[i] for i in id2tok}

corpus = {}
bad = 0
for ln in lines[1:]:
    if not ln.strip():
        continue
    parts = ln.split(" ")
    if len(parts) != 3 or parts[1] not in ("S", "F"):
        bad += 1
        continue
    inst, lab, seq = parts
    toks = []
    ok = True
    for ch in seq:
        if ch not in char2tok:
            ok = False
            break
        toks.append(char2tok[ch])
    if not ok:
        bad += 1
        continue
    corpus[inst] = {"toks": toks, "resolved": lab == "S", "label": lab}

n = len(corpus)
n_res = sum(1 for v in corpus.values() if v["resolved"])
n_steps = sum(len(v["toks"]) for v in corpus.values())
print(f"parsed traces={n}  resolved={n_res} ({100*n_res/max(n,1):.1f}%)  "
      f"failure={n-n_res}  total_steps={n_steps}  malformed_lines={bad}")

(DATA / "corpus.json").write_text(json.dumps(corpus, indent=0))
print(f"wrote {DATA/'corpus.json'}")
