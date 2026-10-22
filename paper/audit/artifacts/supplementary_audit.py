"""Read-only, source-mapped checks supplementing the installed skill checkers.

Run from the repository root with python3 -B. Writes only audit artifacts.
Lexical density is a count against a disclosed lexicon, never authorship evidence.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper"
OUT = Path(__file__).resolve().parent
SCRIPTS = Path("/home/duyennh/.agents/skills/latex-paper-en/scripts")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))
from tex_loader import assemble
from deai_check import AITraceChecker


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def group(text, start):
    assert text[start] == "{"
    depth = 1
    for index in range(start + 1, len(text)):
        if text[index] in "{}" and text[index - 1] != "\\":
            depth += 1 if text[index] == "{" else -1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError("Unbalanced group")


entry = PAPER / "MobilePIXORNeXt.tex"
doc = assemble(entry)
assert not doc.missing, doc.missing
(OUT / "assembled_source.tex.txt").write_text(doc.content)
save("source_map.json", [{"assembled_line": i, "file": f, "source_line": n}
                         for i, (f, n) in enumerate(doc.origins, 1)])
text = uncomment(doc.content)
locations = lambda position: dict(zip(("file", "line"), doc.origin(text[:position].count("\n") + 1)))
labels = collections.defaultdict(list)
for match in re.finditer(r"\\label\{([^}]+)\}", text):
    labels[match[1]].append(locations(match.start()))
refs = collections.defaultdict(list)
for match in re.finditer(r"\\(?:ref|eqref|autoref|[cC]ref|pageref)\*?\{([^}]+)\}", text):
    for key in match[1].split(","):
        refs[key.strip()].append(locations(match.start()))
bib = uncomment((PAPER / "references.bib").read_text())
bib_keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib)
cites = set()
for match in re.finditer(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", text):
    cites.update(key.strip() for key in match[1].split(","))
reference_metrics = {
    "labels": len(labels), "reference_occurrences": sum(map(len, refs.values())),
    "undefined_references": sorted(set(refs) - set(labels)),
    "duplicate_labels": {key: value for key, value in labels.items() if len(value) > 1},
    "unreferenced_labels_all": {key: value for key, value in labels.items() if key not in refs},
    "unreferenced_equation_labels": {key: value for key, value in labels.items()
                                      if key.startswith("eq:") and key not in refs},
    "bib_entries": len(bib_keys), "unique_cited_keys": len(cites),
    "undefined_citations": sorted(cites - set(bib_keys)),
    "uncited_bib_entries": sorted(set(bib_keys) - cites),
    "duplicate_bib_keys": [key for key, count in collections.Counter(bib_keys).items() if count > 1],
}
save("reference_metrics.json", reference_metrics)

raw_abstract = (PAPER / "sections/abstract.tex").read_text().strip()
abstract = uncomment(raw_abstract).replace(r"\%", "%")
abstract = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[A-Za-z]+|[{}]", "", m[1]), abstract)
main = uncomment(entry.read_text())
keywords, _ = group(main, main.index("{", main.index(r"\keywords")))
save("front_matter_metrics.json", {
    "abstract_words_whitespace": len(abstract.split()),
    "abstract_words_lexical": len(re.findall(r"\b[A-Za-z0-9]+(?:[-'/.,][A-Za-z0-9]+)*\b", abstract)),
    "abstract_word_limit": [150, 250], "word_count_rule": "split on whitespace after math rendering and escaped-percent normalization; hyphenated compounds stay intact",
    "abstract_citation_commands": len(re.findall(r"\\cite\w*", raw_abstract)),
    "keywords": [x.strip() for x in keywords.split(r"\and")],
    "keyword_count": len(keywords.split(r"\and")),
    "abstract_uppercase_tokens": sorted(set(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", abstract))),
    "declarations_section_present": r"\section*{Declarations}" in text,
    "official_statements_and_declarations_heading_present": r"\section*{Statements and Declarations}" in text,
    "abstract_rendered_for_count": abstract,
})

checker = AITraceChecker(entry)
analysis = checker.analyze_document()
for result in analysis["sections"].values():
    result["trace_per_source_line_percent"] = checker.calculate_density_score(result)
    for trace in result["traces"]:
        trace["source_file"], trace["source_line"] = checker.doc.origin(trace["line"])
save("deai_api_analysis.json", analysis)
patterns = [
    ("innovative", r"\binnovative\b"), ("pioneering", r"\bpioneering\b"),
    ("revolutionary", r"\brevolutionary\b"), ("transformative", r"\btransformative\b"),
    ("breakthrough", r"\bbreakthrough\b"), ("unprecedented", r"\bunprecedented\b"),
    ("remarkable", r"\bremarkable\b"), ("superior", r"\bsuperior\b"),
    ("surpass", r"\bsurpass(?:es|ed|ing)?\b"), ("state-of-the-art", r"\bstate[- ]of[- ]the[- ]art\b"),
    ("highlights the potential of", r"\bhighlights? the potential of\b"),
    ("pave the way", r"\bpaves? the way\b"),
    ("profound challenge", r"\bprofound challenges?\b"), ("at its essence", r"\bat its essence\b"),
]
sections = collections.defaultdict(list)
skipped = {"equation", "align", "gather", "multline", "eqnarray", "displaymath",
           "figure", "table", "tabular", "algorithm", "algorithmic", "verbatim", "lstlisting"}
active_environment = None
display_math = False
for line, raw in enumerate(checker.doc.lines, 1):
    source_file, source_line = checker.doc.origin(line)
    if not source_file.startswith("sections/") or source_file.endswith("figure_helpers.tex"):
        continue
    raw = uncomment(raw)
    if active_environment:
        if re.search(r"\\end\{" + re.escape(active_environment) + r"\}", raw):
            active_environment = None
        continue
    env = re.search(r"\\begin\{([^}]+)\}", raw)
    if env and env[1].rstrip("*") in skipped:
        if not re.search(r"\\end\{" + re.escape(env[1]) + r"\}", raw):
            active_environment = env[1]
        continue
    # An actual display delimiter is \[, not the \\[5pt] line break in the helper.
    if display_math:
        if re.search(r"(?<!\\)\\\]", raw):
            display_math = False
        continue
    if re.search(r"(?<!\\)\\\[", raw):
        if not re.search(r"(?<!\\)\\\]", raw):
            display_math = True
        continue
    if re.search(r"\\(?:paperfigure|caption|(?:sub)*section|paragraph|label|begin|end|newcommand|renewcommand)\b", raw):
        continue
    if source_file == "sections/methodology.tex" and 409 <= source_line <= 437:
        continue  # The explicitly delimited inference algorithm is not prose.
    visible = checker.parser.extract_visible_text(raw).strip()
    visible = re.sub(r"\\[A-Za-z@]+\*?", " ", visible)
    visible = re.sub(r"[{}]", " ", visible)
    if visible:
        sections[source_file].append((source_line, visible))
density = []
for source in ["abstract", "introduction", "related_work", "methodology", "experiments", "results", "conclusion", "declarations"]:
    path = f"sections/{source}.tex"
    corpus = sections[path]
    visible = " ".join(value for _, value in corpus)
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", visible)
    hits = []
    for line, value in corpus:
        for term, pattern in patterns:
            for match in re.finditer(pattern, value, re.I):
                hits.append({"term": term, "match": match[0], "line": line,
                             "matched_words": len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", match[0]))})
    matched = sum(hit["matched_words"] for hit in hits)
    density.append({"section": source, "file": path, "word_count": len(words),
                    "lexical_hits": len(hits), "matched_word_count": matched,
                    "lexical_density_percent": round(100 * matched / len(words), 6) if words else 0,
                    "hits": hits})
save("lexical_density.json", {"lexicon_source": "paper-audit/references/PRE_SUBMISSION_RULES.md (14 groups)",
                              "formula": "100 * matched lexicon word tokens / visible English word tokens",
                              "exclusions": "comments, citations, inline/display math, figures/tables/captions, algorithm 1, headings, macro definitions",
                              "caveat": "Mechanical lexical statistic; not probability of AI authorship. Separate from deai trace-count/source-line score.",
                              "patterns": dict(patterns), "sections": density})

figures = []
for i, line in enumerate(doc.lines, 1):
    line = uncomment(line)
    for match in re.finditer(r"\\paperfigure(?:\[[^\]]*\])?\{([^}]+)\}", line):
        if "#" in match[1]:
            continue
        source_file, source_line = doc.origin(i)
        target = PAPER / "figures" / match[1]
        figures.append({"file": source_file, "line": source_line, "asset": match[1],
                        "exists": target.is_file(), "bytes": target.stat().st_size if target.exists() else None})
save("figure_metrics.json", {"actual_figure_calls": len(figures),
                             "missing_actual_assets": [x for x in figures if not x["exists"]], "figures": figures})

units = []
for i, raw in enumerate(doc.lines, 1):
    line = uncomment(raw)
    for match in re.finditer(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?:ms|MB|GB|FPS|Hz|km|cm|mm)\b", line):
        f, n = doc.origin(i)
        units.append({"file": f, "line": n, "quote": match[0]})
save("number_unit_metrics.json", {"attached_physical_unit_candidates": units,
                                  "note": "Only number-unit adjacency is deterministic here; percent vs percentage-point semantics are reviewed separately."})
result_lines = (PAPER / "sections/results.tex").read_text().splitlines()
epoch_candidates = []
for number, line in enumerate(result_lines, 1):
    if "checkpoint selected by validation AP from the extended run" in line:
        match = re.search(r"extended run \(Epoch (\d+)\)", line)
        if match:
            epoch_candidates.append({"role": "prose", "line": number, "epoch": int(match[1]), "quote": match[0]})
    if "validation AP maximization (Epoch" in line:
        match = re.search(r"validation AP maximization \(Epoch (\d+)\)", line)
        epoch_candidates.append({"role": "caption", "line": number, "epoch": int(match[1]), "quote": match[0]})
    match = re.search(r"\\textbf\{Best AP\}\s*&\s*\\textbf\{(\d+)\}", line)
    if match:
        epoch_candidates.append({"role": "table_row", "line": number, "epoch": int(match[1]), "quote": match[0]})
save("checkpoint_consistency.json", {"file": "sections/results.tex", "records": epoch_candidates,
                                      "consistent": len({x["epoch"] for x in epoch_candidates}) <= 1})
before = json.loads((OUT / "source_hashes_before.json").read_text())
after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in before}
save("source_hashes_after.json", after)
changed = [p for p in before if before[p] != after[p]]
save("source_integrity.json", {"checked_files": len(before), "changed_files": changed})
print(json.dumps({"references": reference_metrics, "lexical_density": density,
                  "figure_calls": len(figures), "attached_units": units, "changed_sources": changed}, indent=2))
