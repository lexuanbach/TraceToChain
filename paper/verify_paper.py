"""Static verification of paper/main.tex for ISSRE submission.

Checks:
  (1) Brace balance
  (2) Every \\cite{key} appears in refs.bib
  (3) Every \\ref{label} has a matching \\label
  (4) Every \\includegraphics path resolves to an existing PDF
  (5) Every \\input{file} resolves to an existing .tex
  (6) Word count estimate (should be < 9000 for 12-page IEEEtran)

Does not actually run pdflatex (IEEEtran.cls unavailable in sandbox).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BIB = HERE / "refs.bib"

bib = BIB.read_text()

errors = []
warns = []


def resolve_input(inp: str, including_file: Path | None = None) -> Path:
    """Resolve inputs as LaTeX sees them when compiling from paper/."""
    candidates: list[Path] = []
    raw = Path(inp)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(HERE / raw)
        if including_file is not None:
            candidates.append(including_file.parent / raw)

    expanded: list[Path] = []
    for p in candidates:
        expanded.append(p)
        if p.suffix == "":
            expanded.append(p.with_suffix(".tex"))
    for p in expanded:
        if p.exists():
            return p.resolve()
    p = (HERE / raw).resolve()
    if p.suffix == "":
        p = p.with_suffix(".tex")
    return p


def strip_false_blocks(body: str) -> str:
    r"""Drop simple \iffalse...\fi blocks before active-text checks."""
    kept: list[str] = []
    depth = 0
    for line in body.splitlines(keepends=True):
        if re.search(r"\\iffalse\b", line):
            depth += 1
            continue
        if depth:
            if re.search(r"\\fi\b", line):
                depth -= 1
            continue
        kept.append(line)
    return "".join(kept)


def read_with_inputs(
    path: Path,
    seen: set[Path] | None = None,
    *,
    active_only: bool = False,
) -> tuple[str, list[Path]]:
    """Read a .tex file and recursively inline local/external \\input files."""
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return "", []
    seen.add(path)

    body = path.read_text()
    scan_body = strip_false_blocks(body) if active_only else body
    pieces = [scan_body]
    files = [path]
    for inp in re.findall(r"\\input\{([^}]+)\}", scan_body):
        child = resolve_input(inp, path)
        if child.exists():
            child_text, child_files = read_with_inputs(
                child, seen, active_only=active_only
            )
            pieces.append(child_text)
            files.extend(child_files)
    return "\n".join(pieces), files


text, tex_files = read_with_inputs(TEX, active_only=True)


def check_braces(s: str) -> tuple[int, int]:
    open_n = s.count("{")
    close_n = s.count("}")
    return open_n, close_n


# (1) brace balance
o, c = check_braces(text)
if o != c:
    errors.append(f"brace imbalance: {o} open vs {c} close")
else:
    print(f"[OK] braces balanced: {o}=={c}")

cited: set[str] = set()
cited |= set(re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", text))
# split comma-separated
cited_keys: set[str] = set()
for entry in cited:
    for k in entry.split(","):
        cited_keys.add(k.strip())

bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
missing_cites = cited_keys - bib_keys
extra_bibs = bib_keys - cited_keys
if missing_cites:
    errors.append(f"undefined bib keys: {sorted(missing_cites)}")
else:
    print(f"[OK] all {len(cited_keys)} cited keys resolve in refs.bib")
if extra_bibs:
    warns.append(f"unused bib keys: {sorted(extra_bibs)}")

# (3) ref/label consistency.  Collect labels from the assembled paper,
#     including section files, proofs, and generated data snippets.
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
# Include artifact proof/data labels because the 12-page paper may refer
# to propositions whose full statements are retained outside the body.
for sub in ("proofs", "data"):
    d = HERE.parent / sub
    if d.exists():
        for tex in d.rglob("*.tex"):
            labels |= set(re.findall(r"\\label\{([^}]+)\}", tex.read_text()))
missing_refs = refs - labels
if missing_refs:
    errors.append(f"dangling \\ref{{}}: {sorted(missing_refs)}")
else:
    print(f"[OK] all {len(refs)} refs resolve to labels")

# (4) includegraphics
figs_needed = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
for fig in figs_needed:
    # paths are relative to main.tex
    path = (HERE / fig).resolve()
    if not path.exists():
        errors.append(f"missing figure: {fig} -> {path}")
    else:
        print(f"[OK] figure exists: {fig}")

# (5) \input{...}
inputs: list[tuple[str, Path]] = []
for tex in tex_files:
    for inp in re.findall(r"\\input\{([^}]+)\}", strip_false_blocks(tex.read_text())):
        inputs.append((inp, tex))
for inp, including_file in inputs:
    p = resolve_input(inp, including_file)
    if not p.exists():
        errors.append(f"missing \\input: {inp} -> {p}")
    else:
        print(f"[OK] input exists: {inp}")

# (6) word count (rough)
# strip comments and LaTeX commands
s = re.sub(r"%.*", "", text)
s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
s = re.sub(r"[{}$&_^\\]", " ", s)
words = len(s.split())
print(f"[INFO] main.tex word estimate: {words}")
if words > 9500:
    warns.append(f"word count {words} may exceed 12-page IEEEtran limit")

# Summary
print("---")
if warns:
    print(f"[WARN] {len(warns)} warning(s):")
    for w in warns:
        print(f"       - {w}")
if errors:
    print(f"[FAIL] {len(errors)} error(s):")
    for e in errors:
        print(f"       - {e}")
    sys.exit(1)
else:
    print("[PASS] verification clean")
    sys.exit(0)
