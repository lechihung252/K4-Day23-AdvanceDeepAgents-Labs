"""Offline tests (no network, no LLM):   pip install pytest && python -m pytest -q"""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research  # noqa: E402
import tools  # noqa: E402
from check_citations import check  # noqa: E402
from finalize_citations import finalize  # noqa: E402

SOURCES = [{"n": 1, "id": "a", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025", "source": "arxiv"},
           {"n": 2, "id": "b", "url": "https://huggingface.co/papers/2502.1", "title": "B", "date": "", "source": "hf-search"}]
GOOD = ("# T\n\nClaim [1]. Other [2].\n\n## References\n"
        "[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025)\n[2] B. hf-search. https://huggingface.co/papers/2502.1\n")


# ---- with_retry ----
@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.time, "sleep", calls.append)
    return calls


def test_retry_succeeds_after_transient_errors(sleeps):
    outcomes = iter([tools.RetryableError("x"), tools.RetryableError("x"), "ok"])

    def fn():
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value
    assert tools.with_retry(fn, attempts=5, base=1, cap=30) == "ok"
    assert len(sleeps) == 2 and 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 3


def test_retry_last_attempt_raises_without_sleeping(sleeps):
    def fn():
        raise tools.RetryableError("down")
    with pytest.raises(tools.RetryableError):
        tools.with_retry(fn, attempts=3)
    assert len(sleeps) == 2


def test_retry_after_is_used_and_capped(sleeps):
    outcomes = iter([tools.RetryableError("x", retry_after=7), tools.RetryableError("x", retry_after=999), "ok"])

    def fn():
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value
    tools.with_retry(fn, attempts=5, cap=30)
    assert sleeps == [7, 30]


def test_other_errors_are_not_retried(sleeps):
    def fn():
        raise KeyError("bug")
    with pytest.raises(KeyError):
        tools.with_retry(fn)
    assert sleeps == []


def test_status_codes_become_retryable(monkeypatch):
    response = httpx.Response(503, headers={"Retry-After": "4"}, request=httpx.Request("GET", "https://x"))
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: response)
    with pytest.raises(tools.RetryableError) as info:
        tools._request("GET", "https://x")
    assert info.value.retry_after == 4


# ---- tools ----
def test_arxiv_query_is_sanitised():
    assert tools._arxiv_terms('all:"world model" AND ti:survey') == ["world", "model", "survey"]
    assert tools.arxiv_search.invoke({"query": "\"' : AND"}) == "NO RESULTS"


def test_exa_rate_limit_flag_is_retried_and_key_redacted(monkeypatch, sleeps):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    limited = {"result": {"content": [{"type": "text", "text": "You've hit Exa's rate limit"}],
                          "_meta": {"rateLimited": True}}}

    def fake_request(method, url, **kwargs):
        assert "secret-key-123" not in url
        raise RuntimeError("connection failed for key secret-key-123")
    monkeypatch.setattr(tools, "_request", fake_request)
    assert "secret-key-123" not in tools.web_search.invoke({"query": "x"})

    sse = "event: message\ndata: " + json.dumps(limited) + "\n\n"
    response = httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    monkeypatch.setattr(tools, "_request", lambda *a, **k: response)
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR") and len(sleeps) == 5   # retried, never returned the notice as content


def test_only_transient_model_errors_are_retried():
    import agents
    from google.genai import errors
    assert agents._transient_model_error(errors.ServerError(503, {"error": {"message": "high demand"}}))
    assert agents._transient_model_error(errors.ClientError(429, {"error": {"message": "quota"}}))
    assert not agents._transient_model_error(errors.ClientError(400, {"error": {"message": "bad request"}}))
    daily = {"error": {"message": "Quota exceeded", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"},
                                                                {"retryDelay": "53742s"}]}}
    assert not agents._transient_model_error(errors.ClientError(429, daily))
    assert not agents._transient_model_error(KeyError("bug"))


# ---- check_citations ----
def test_check_accepts_valid_report():
    assert check(GOOD, SOURCES) == []


def test_check_accepts_grouped_citations():
    assert check(GOOD.replace("Claim [1]. Other [2].", "Both [1-2]."), SOURCES) == []


@pytest.mark.parametrize("report, sources, expected", [
    (GOOD, [], "no sources"),
    (GOOD.replace("## References", "## Refs"), SOURCES, "no '## References'"),
    (GOOD.replace("Other [2]", "Other [3]"), SOURCES, "[3] cited but missing"),
    (GOOD.replace("Other [2]", "Other"), SOURCES, "source [2] never cited"),
    (GOOD.replace("[2] B. hf-search. https://huggingface.co/papers/2502.1\n", ""), SOURCES, "source [2] has no line"),
    (GOOD.replace("(2025)", "; B https://huggingface.co/papers/2502.1"), SOURCES, "exactly one URL"),
    (GOOD.replace("abs/2501.00001 ", "abs/9999 "), SOURCES, "differs from sources.json"),
    (GOOD + "[2] again https://huggingface.co/papers/2502.1\n", SOURCES, "more than once"),
    (GOOD, [dict(SOURCES[0]), dict(SOURCES[1], url=SOURCES[0]["url"])], "duplicates"),
    (GOOD, [dict(SOURCES[0], n="1"), SOURCES[1]], "not an integer"),
    (GOOD, [dict(SOURCES[0], url="ftp://x"), SOURCES[1]], "does not start with http"),
])
def test_check_finds_problems(report, sources, expected):
    assert any(expected in p for p in check(report, sources)), check(report, sources)


def test_citations_in_code_and_links_are_ignored():
    report = GOOD.replace("Other [2].", "Other [2]. `x[5]` and [7](https://e.org).")
    assert check(report, SOURCES) == []


def test_finalizer_output_passes_validator():
    body = "# T\n\nB first [2], then A [1, 2]."
    report, sources, problems = finalize(body, SOURCES)
    assert problems == [] and check(report, sources) == []


# ---- research.py ----
@pytest.mark.parametrize("topic, slug", [("survey about world model", "survey-about-world-model"),
                                         ("../../x", "x"), ("", "topic"), ("!!!", "topic")])
def test_slugify(topic, slug):
    assert research.slugify(topic) == slug


def test_slugify_is_capped():
    assert len(research.slugify("a b " * 100)) <= 60


class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        from deepagents.backends.protocol import FileDownloadResponse
        return [FileDownloadResponse(path=p, content=self.files.get(p), error=None) for p in paths]


def test_save_outputs_writes_three_files(tmp_path):
    from langchain_core.messages import AIMessage
    messages = [AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": str(i)} for i in range(3)],
                          usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})]
    backend = FakeBackend({research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: json.dumps(SOURCES).encode()})
    path = research.save_outputs(backend, "My Topic", messages, 12.34, "m", reports_dir=tmp_path)
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text())
    assert path.read_text() == GOOD and (tmp_path / "my-topic.sources.json").exists()
    assert meta["subagent_calls"] == 3 and meta["source_families"] == ["arxiv", "hf-search"]
    assert meta["tokens"] == {"input": 10, "output": 5} and meta["elapsed_s"] == 12.3


@pytest.mark.parametrize("files", [{}, {research.REPORT_PATH: b"  "},
                                   {research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: b"{bad"}])
def test_failed_run_writes_nothing(tmp_path, files):
    with pytest.raises(RuntimeError):
        research.save_outputs(FakeBackend(files), "t", [], 1.0, "m", reports_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_main_without_topic_returns_2():
    assert research.main("  ") == 2


def test_tool_exception_becomes_error_message():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool as make_tool

    import agents

    @make_tool
    def write_file(file_path: str) -> str:
        """Write a file."""
        raise RuntimeError(f"Failed to upload files: 400: {file_path}: permission denied")

    class ToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/x.md"}, "id": "1"}]),
        AIMessage(content="fixed")]))
    out = create_agent(model, tools=[write_file], middleware=agents._limits(5, 5)).invoke({"messages": [("user", "go")]})
    tool_result = next(m for m in out["messages"] if m.type == "tool")
    assert tool_result.content.startswith("ERROR: RuntimeError") and out["messages"][-1].content == "fixed"


def test_check_rejects_url_not_matching_family():
    sources = [dict(SOURCES[0], url="https://arxiv.science/abs/2501.00001"), SOURCES[1]]
    report = GOOD.replace("https://arxiv.org/abs/2501.00001", "https://arxiv.science/abs/2501.00001")
    assert any("labelled 'arxiv'" in p for p in check(report, sources))
    assert any("must be one of" in p for p in check(GOOD, [dict(SOURCES[0], source="paper"), SOURCES[1]]))


def test_sources_must_come_from_the_notes(tmp_path):
    from check_citations import check_against_notes
    (tmp_path / "01-a.md").write_text(f"## A\n- url: {SOURCES[0]['url']}\n")
    problems = check_against_notes(SOURCES, str(tmp_path))
    assert len(problems) == 1 and SOURCES[1]["url"] in problems[0]


def test_check_flags_uncited_paragraphs():
    report = GOOD.replace("Claim [1]. Other [2].", "Claim [1]. Other [2].\n\n" + "word " * 15 + ".")
    assert any("uncited text" in p for p in check(report, SOURCES))


def test_check_families_needs_three():
    from check_citations import check_families
    assert check_families(SOURCES) and "only 2 source families" in check_families(SOURCES)[0]
    assert check_families(SOURCES + [dict(SOURCES[0], n=3, source="web", url="https://e.org")]) == []


def test_check_structure():
    from check_citations import check_structure
    sections = "## TL;DR\n## Background\n## A\n## B\n## C\n## Trends and open problems\n## References\n"
    assert check_structure(sections) == []
    assert any("2 theme sections" in p for p in check_structure(sections.replace("## C\n", "")))
    assert any("missing section '## TL;DR'" in p for p in check_structure(sections.replace("## TL;DR\n", "")))
