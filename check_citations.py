"""check_citations.py - Citation validator.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
Beyond the citation rules it checks that every url matches its source family and, inside the sandbox, that every
source url appears verbatim in a researcher notes file (the lead may not invent or edit urls).
"""
import glob
import json
import os
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
NOTES_DIR = "/tmp/work/research/notes"
FAMILY_URL = {"arxiv": "https://arxiv.org/abs/", "hf-daily": "https://huggingface.co/papers/",
              "hf-search": "https://huggingface.co/papers/", "web": ""}

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not a Markdown link [3](url)
_REF_LINE = re.compile(r"^\s*\[(\d+)\]")
_URL = re.compile(r"https?://[^\s<>]+")
UNCITED_WORDS = 12   # a body paragraph or bullet this long must carry at least one [n]


def _numbers(group):
    """'1, 3-5' -> [1, 3, 4, 5]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited(body):
    """Numbers cited as [n] in the body, ignoring code spans/blocks."""
    cited = set()
    for match in _CITATION.finditer(_CODE.sub(" ", body)):
        cited.update(_numbers(match.group(1)))
    return cited


def _same_url(found, url):
    """A URL found in text may carry trailing punctuation ("...abs/1234.")."""
    return found == url or found.rstrip(".,;:)]") == url.rstrip(".,;:)]")


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems = []
    by_n, seen_urls = {}, {}
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources.json entry #{i + 1} is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"sources.json entry #{i + 1}: n={n!r} is not an integer")
            continue
        if n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}]: url {url!r} does not start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"source [{n}]: url {url} duplicates source [{seen_urls[url]}]")
        else:
            seen_urls[url] = n
        family = entry.get("source")
        if family not in FAMILY_URL:
            problems.append(f"source [{n}]: source {family!r} must be one of {', '.join(FAMILY_URL)}")
        elif isinstance(url, str) and not url.startswith(FAMILY_URL[family]):
            problems.append(f"source [{n}] is labelled {family!r} but its url {url} does not start with "
                            f"{FAMILY_URL[family]} (label it with the tool that returned it, e.g. 'web')")

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        return problems + ["the report has no '## References' heading"]
    if len(headings) > 1:
        problems.append("the report has more than one '## References' heading")
    body, references = report_text[:headings[-1].start()], report_text[headings[-1].end():]

    cited = _cited(body)
    for line in _CODE.sub(" ", body).splitlines():
        text = line.strip()
        if text and not text.startswith("#") and len(text.split()) >= UNCITED_WORDS and not _cited(text):
            problems.append(f"uncited text in the body: \"{text[:70]}...\" (every claim needs a [n]: cite a source "
                            "from the notes, or remove the sentence)")
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys())]
    problems += [f"source [{n}] never cited in the report body" for n in sorted(by_n.keys() - cited)]

    ref_lines = {}
    for line in references.splitlines():
        match = _REF_LINE.match(line)
        if not match:
            if _URL.search(line):
                problems.append(f"reference line without a [n] number: {line.strip()[:80]}")
            continue
        n = int(match.group(1))
        if n in ref_lines:
            problems.append(f"reference [{n}] is listed more than once")
            continue
        ref_lines[n] = line
        if n not in by_n:
            problems.append(f"reference [{n}] is not a source in sources.json")
            continue
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must hold exactly one URL, found {len(urls)} (one source per line)")
        elif not _same_url(urls[0], str(by_n[n].get("url"))):
            problems.append(f"reference [{n}] URL {urls[0]} differs from sources.json url {by_n[n].get('url')}")
    problems += [f"source [{n}] has no line in '## References'" for n in sorted(by_n.keys() - ref_lines.keys())]
    return problems


def check_against_notes(sources, notes_dir=NOTES_DIR):
    """Every source url must appear verbatim in some researcher notes file."""
    notes = "\n".join(open(path, encoding="utf-8", errors="replace").read()
                      for path in glob.glob(os.path.join(notes_dir, "*.md")))
    return [f"source [{e.get('n')}] url {e.get('url')} appears in no notes file under {notes_dir}: copy urls from "
            "the notes exactly, never invent or edit them"
            for e in sources if isinstance(e, dict) and str(e.get("url")) not in notes]


def check_families(sources, minimum=3):
    """RUBRIC 2.2: the report must draw on at least `minimum` source families."""
    families = {e.get("source") for e in sources if isinstance(e, dict)} & FAMILY_URL.keys()
    if len(families) >= minimum:
        return []
    return [f"only {len(families)} source families ({', '.join(sorted(families)) or 'none'}): cite sources of at least "
            f"{minimum} of {', '.join(FAMILY_URL)} (take one from the notes, or delegate a researcher to a missing family)"]


def check_structure(report_text):
    """REPORT_TEMPLATE.md: the required sections, with 3 to 6 theme sections between Background and Trends."""
    headings = [h.strip() for h in re.findall(r"(?m)^##[ \t]+(.+)$", report_text)]
    required = ["TL;DR", "Background", "Trends and open problems", "References"]
    problems = [f"missing section '## {h}'" for h in required if h.lower() not in (x.lower() for x in headings)]
    themes = [h for h in headings if h.lower() not in (r.lower() for r in required)]
    if not 3 <= len(themes) <= 6:
        problems.append(f"{len(themes)} theme sections ({'; '.join(themes) or 'none'}): write 3 to 6 '## <Theme>' "
                        "sections between Background and Trends and open problems")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if os.path.isdir(NOTES_DIR) and isinstance(sources, list):  # inside the sandbox, where the notes live
        problems += check_against_notes(sources) + check_families(sources) + check_structure(report)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
