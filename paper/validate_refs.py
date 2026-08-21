"""Validate active bibliography entries used by the paper.

This checker is intentionally lightweight.  It does not replace a publisher or
DBLP audit, but it catches the bibliography problems that most often survive
manual editing: missing entries, ``and others`` author lists, missing stable
identifiers, duplicated DOI/eprint identifiers, and entry-type mismatches.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BIB = HERE / "refs.bib"


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


def resolve_input(inp: str, including_file: Path | None = None) -> Path:
    raw = Path(inp)
    candidates: list[Path] = []
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

    fallback = (HERE / raw).resolve()
    return fallback if fallback.suffix else fallback.with_suffix(".tex")


def read_with_inputs(path: Path, seen: set[Path] | None = None) -> str:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    body = strip_false_blocks(path.read_text())
    pieces = [body]
    for inp in re.findall(r"\\input\{([^}]+)\}", body):
        child = resolve_input(inp, path)
        if child.exists():
            pieces.append(read_with_inputs(child, seen))
    return "\n".join(pieces)


def parse_bib_entries(bib: str) -> dict[str, dict[str, str]]:
    """Parse enough BibTeX for validation of this repository's refs.bib."""
    entries: dict[str, dict[str, str]] = {}
    i = 0
    while True:
        match = re.search(r"@(\w+)\s*\{\s*([^,]+)\s*,", bib[i:], re.S)
        if not match:
            break
        kind = match.group(1).lower()
        key = match.group(2).strip()
        start = i + match.end()
        depth = 1
        j = start
        while j < len(bib) and depth:
            if bib[j] == "{":
                depth += 1
            elif bib[j] == "}":
                depth -= 1
            j += 1
        body = bib[start : j - 1]
        fields = {"ENTRYTYPE": kind, "raw": body}
        for field in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*,?", body, re.S):
            fields[field.group(1).lower()] = re.sub(r"\s+", " ", field.group(2)).strip()
        entries[key] = fields
        i = j
    return entries


def cite_keys(text: str) -> set[str]:
    raw = re.findall(r"\\cite(?:[tp])?(?:\[[^\]]*\])?\{([^}]+)\}", text)
    keys: set[str] = set()
    for group in raw:
        keys |= {k.strip() for k in group.split(",") if k.strip()}
    return keys


def main() -> int:
    text = read_with_inputs(TEX)
    cited = cite_keys(text)
    entries = parse_bib_entries(BIB.read_text())
    errors: list[str] = []
    warns: list[str] = []

    missing = cited - set(entries)
    if missing:
        errors.append(f"missing bib entries for active citations: {sorted(missing)}")

    active_entries = {k: entries[k] for k in sorted(cited & set(entries))}
    doi_seen: dict[str, str] = {}
    eprint_seen: dict[str, str] = {}

    for key, entry in active_entries.items():
        raw = entry["raw"]
        kind = entry["ENTRYTYPE"]
        title = entry.get("title", "")
        has_stable_id = any(entry.get(f) for f in ("doi", "url", "eprint"))

        if re.search(r"\band\s+others\b", raw, re.I):
            errors.append(f"{key}: active entry still uses 'and others'")

        if not has_stable_id and kind not in {"book", "manual"}:
            errors.append(f"{key}: non-book entry lacks doi, url, or eprint")

        if kind == "article" and "journal" not in entry:
            errors.append(f"{key}: @article entry lacks a journal field")
        if kind == "inproceedings" and "journal" in entry:
            errors.append(f"{key}: @inproceedings entry has a journal field")
        if kind == "book" and "publisher" not in entry:
            errors.append(f"{key}: @book entry lacks a publisher field")

        doi = entry.get("doi")
        if doi:
            if doi in doi_seen:
                errors.append(f"{key}: duplicate DOI with {doi_seen[doi]} ({doi})")
            doi_seen[doi] = key

        eprint = entry.get("eprint")
        if eprint:
            if eprint in eprint_seen:
                errors.append(f"{key}: duplicate eprint with {eprint_seen[eprint]} ({eprint})")
            eprint_seen[eprint] = key
            if entry.get("archiveprefix", "").lower() == "arxiv":
                arxiv_doi = f"10.48550/arXiv.{eprint}"
                if doi and doi != arxiv_doi:
                    warns.append(f"{key}: arXiv eprint has non-arXiv DOI {doi}")
                if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", eprint):
                    errors.append(f"{key}: arXiv eprint does not look canonical ({eprint})")

        if not re.search(r"\{LLMs?\}", title) and re.search(r"\bLLMs?\b", title):
            warns.append(f"{key}: title may need braces around LLM")
        if "{PCTL}" not in title and "PCTL" in title:
            warns.append(f"{key}: title may need braces around PCTL")

    unused = set(entries) - cited
    if unused:
        warns.append(f"unused bib keys: {sorted(unused)}")

    print(f"[OK] checked {len(active_entries)} active bibliography entries")
    print(f"[OK] active entries with DOI: {sum(1 for e in active_entries.values() if e.get('doi'))}")
    print(f"[OK] active entries with arXiv eprint: {sum(1 for e in active_entries.values() if e.get('eprint'))}")
    print("---")
    if warns:
        print(f"[WARN] {len(warns)} warning(s):")
        for warning in warns:
            print(f"       - {warning}")
    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for error in errors:
            print(f"       - {error}")
        return 1
    print("[PASS] reference validation clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
