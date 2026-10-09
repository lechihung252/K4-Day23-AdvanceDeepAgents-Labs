"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RECURSION_LIMIT = 1000   # LangGraph step cap for the lead graph (~2 steps per model->tool turn); subagents are capped
                         # by their own middleware limits in agents.py


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[\W_]+", "-", topic.lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n"
            f"Today's date: {time.strftime('%Y-%m-%d')} (\"the last two years\" counts back from it).\n"
            "Produce the cited survey report by following your workflow step by step, and finish only when the "
            "validator prints OK.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    tool_calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": tool_calls["task"],
            "tool_calls": dict(tool_calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, usage_all=None):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A missing/empty report or a missing/invalid sources.json raises RuntimeError and writes nothing."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, sources_raw = files.get(REPORT_PATH) or b"", files.get(SOURCES_PATH) or b""
    if not report.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} is missing or empty)")
    try:
        sources = json.loads(sources_raw)
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is missing or not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} is not a non-empty JSON list")
    problems = check(report.decode("utf-8", errors="replace"), sources)
    if problems:  # saved anyway so the run can be inspected; self_check.py will flag it
        print("WARNING: the report does not pass check_citations:\n  " + "\n  ".join(problems[:10]), file=sys.stderr)

    families = sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": families}
    if usage_all:  # every model call of the run, lead AND subagents (the real cost; "tokens" is the lead only)
        meta["tokens_all"] = {"input": sum(u.get("input_tokens", 0) for u in usage_all.values()),
                              "output": sum(u.get("output_tokens", 0) for u in usage_all.values())}
    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{slug}.sources.json").write_bytes(sources_raw)   # byte-for-byte what the sandbox produced
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def run_agent(agent, topic):
    """Stream the lead agent, log its tool calls, and return every message it produced (deduplicated by id, so
    messages compacted away by context summarisation are still counted in meta.json)."""
    seen = {}
    inputs = {"messages": [{"role": "user", "content": build_prompt(topic)}]}
    for state in agent.stream(inputs, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values"):
        for message in state.get("messages", []):
            key = message.id or id(message)
            if key in seen:
                continue
            seen[key] = message
            if message.type == "tool" and message.name == "execute":  # finalizer/validator verdicts
                print(f"         -> {' | '.join(str(message.content).splitlines()[:3])[:200]}", file=sys.stderr)
            for call in getattr(message, "tool_calls", None) or []:
                args = call["args"]
                detail = (f"{args.get('subagent_type')}: {args.get('description', '')}" if call["name"] == "task"
                          else args.get("command") or args.get("file_path") or "")
                print(f"  [lead] {call['name']} {' '.join(str(detail).split())[:110]}", file=sys.stderr)
    return list(seen.values())


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL")
        start = time.monotonic()
        with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            with get_usage_metadata_callback() as usage:
                messages = run_agent(build_lead_agent(backend, model), topic)
            report_path = save_outputs(backend, topic, messages, time.monotonic() - start, str(model_name),
                                       usage_all=usage.usage_metadata)
    except Exception as exc:  # noqa: BLE001  (any failure: no report written, non-zero exit)
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
