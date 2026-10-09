"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()  # EXA_API_KEY, also when this file runs on its own (python tools.py)

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

TIMEOUT = httpx.Timeout(60.0, connect=15.0)
RETRY_STATUS = {429, 500, 502, 503, 504}
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_MIN_INTERVAL = 3.0       # arXiv API etiquette: at least 3 s between calls
EXA_QUOTA_EXHAUSTED_S = 600    # a Retry-After this long means "quota used up", not "wait a moment": give up at once
SUMMARY_CHARS = 600
PAGE_CHARS = 12000


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls).

    The wait is the server's Retry-After when given, else exponential backoff base * 2**attempt plus random jitter;
    both are capped at `cap` seconds. The last failure is re-raised without sleeping. Other exceptions propagate.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * 2 ** attempt + random.uniform(0, base), cap)
            time.sleep(delay)
    raise ValueError("attempts must be >= 1")


def _retry_after(response):
    """Seconds from a Retry-After header (delta-seconds or HTTP date), else None."""
    value = (response.headers.get("retry-after") or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _request(method, url, **kwargs):
    """One HTTP call. Transient failures (network errors, 429, 5xx) become RetryableError."""
    try:
        response = httpx.request(method, url, timeout=TIMEOUT, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRY_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", _retry_after(response))
    response.raise_for_status()
    return response


def _redact(text):
    key = (os.getenv("EXA_API_KEY") or "").strip()
    return text.replace(key, "***") if key else text


def _error(exc):
    return _redact(f"ERROR: {type(exc).__name__}: {exc}")


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_CHARS):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()   # researchers run in parallel threads: serialise arXiv calls to keep the spacing
_arxiv_last_call = 0.0
ARXIV_COOLDOWN_S = 600           # after arXiv keeps answering 429 through all retries, skip it for a while
_arxiv_blocked_until = 0.0


def _arxiv_get(params):
    global _arxiv_last_call
    with _arxiv_lock:
        wait = _arxiv_last_call + ARXIV_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return _request("GET", ARXIV_URL, params=params)
        finally:
            _arxiv_last_call = time.monotonic()


def _arxiv_terms(query):
    """LLM input -> plain search terms: drop field prefixes (all:, ti:), boolean operators, quotes and punctuation."""
    query = re.sub(r"\b(?:all|ti|abs|au|cat|co|jr|rn|id):", " ", query, flags=re.IGNORECASE)
    terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
    return [t for t in terms if t.upper() not in {"AND", "OR", "NOT", "ANDNOT"}][:6]


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Use 2-5 short keywords (e.g. "world model video"); every
    keyword must match, so long queries return nothing. Returns a JSON list of {id, url, published, title, summary}
    where url is https://arxiv.org/abs/<id>; sources found here have source family "arxiv"."""
    global _arxiv_blocked_until
    terms = _arxiv_terms(query)
    if not terms:
        return "NO RESULTS"
    if time.monotonic() < _arxiv_blocked_until:
        return "ERROR: arXiv is rate limiting this network right now; use hf_search_papers or web_search instead"
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": _clamp(max_results, 1, 30), "start": 0}
    try:
        try:
            response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        except RetryableError:
            _arxiv_blocked_until = time.monotonic() + ARXIV_COOLDOWN_S
            raise
        records = []
        for entry in ET.fromstring(response.text).findall("a:entry", ATOM):
            raw_id = entry.findtext("a:id", "", ATOM)
            if "/abs/" not in raw_id:  # arXiv reports query errors as an entry without /abs/
                continue
            arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
            records.append({"id": arxiv_id, "url": f"https://arxiv.org/abs/{arxiv_id}",
                            "published": entry.findtext("a:published", "", ATOM)[:10],
                            "title": _clean(entry.findtext("a:title", "", ATOM)),
                            "summary": _short(entry.findtext("a:summary", "", ATOM))})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001  (a tool never raises)
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    paper = item.get("paper") or {}
    paper_id = paper.get("id")
    if not paper_id:
        return None
    return {"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _short(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
            "upvotes": paper.get("upvotes") or 0, "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0}


def _hf_records(url, params):
    items = with_retry(lambda: _request("GET", url, params=params)).json()
    return [r for r in map(_hf_record, items if isinstance(items, list) else []) if r]


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary (all its words must appear); there is no topic search on this endpoint (use
    hf_search_papers for a topic). url is https://huggingface.co/papers/<id>; source family "hf-daily"."""
    date = date.strip()
    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return "ERROR: date must be YYYY-MM-DD"
    params = {"limit": _clamp(limit, 1, 100), **({"date": date} if date else {})}
    try:
        records = _hf_records(HF_DAILY_URL, params)
        words = keyword.lower().split()
        if words:
            records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (semantic search over arXiv papers listed on Hugging Face, with community
    upvotes and GitHub links). Returns a JSON list of {id, url, published, title, summary, upvotes, github, stars}.
    url is https://huggingface.co/papers/<id>; source family "hf-search"."""
    if not query.strip():
        return "NO RESULTS"
    try:
        records = _hf_records(HF_SEARCH_URL, {"q": query.strip(), "limit": _clamp(limit, 1, 50)})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _looks_rate_limited(text):
    return "rate limit" in str(text).lower()


def _meta_rate_limited(meta):
    """True when Exa's result._meta carries a rate-limit flag (any key mentioning rate limit / throttling)."""
    if not isinstance(meta, dict):
        return False
    for key, value in meta.items():
        name = key.lower().replace("_", "").replace("-", "")
        if isinstance(value, dict) and _meta_rate_limited(value):
            return True
        if any(word in name for word in ("ratelimit", "throttl")):
            if value is True or (isinstance(value, str) and any(w in value.lower() for w in ("limit", "exceed"))):
                return True
    return False


def _rpc_message(response):
    """The JSON-RPC answer, from a plain JSON body or from the `data:` lines of a server-sent-events body."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    message = None
    for line in response.text.splitlines():
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if "result" in data or "error" in data:
                message = data
    if message is None:
        raise ValueError("Exa answered without a JSON-RPC result")
    return message


def _exa_once(name, arguments):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:  # sent as a header, not in the URL, so it cannot show up in httpx error messages
        headers["Authorization"] = f"Bearer {key}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    try:
        response = _request("POST", EXA_URL, json=payload, headers=headers)
    except RetryableError as exc:
        if exc.retry_after is not None and exc.retry_after > EXA_QUOTA_EXHAUSTED_S:
            raise RuntimeError(f"Exa quota exhausted (retry after {exc.retry_after:.0f}s); set EXA_API_KEY") from exc
        raise
    message = _rpc_message(response)
    if "error" in message:
        error = message["error"] or {}
        if _looks_rate_limited(error.get("message")):
            raise RetryableError("Exa rate limited (JSON-RPC error)")
        raise RuntimeError(f"Exa JSON-RPC error {error.get('code')}: {str(error.get('message'))[:300]}")
    result = message.get("result") or {}
    text = "\n".join(part.get("text", "") for part in result.get("content") or [] if part.get("type") == "text")
    # The free tier signals "rate limited" with HTTP 200, a short notice as the text and a flag in result._meta.
    if _meta_rate_limited(result.get("_meta")) or (_looks_rate_limited(text) and (result.get("isError")
                                                                                 or len(text) < 600)):
        raise RetryableError("Exa rate limited (result._meta)")
    if result.get("isError"):
        raise RuntimeError(f"Exa tool error: {text[:300]}")
    return text.strip()


def _exa_call(name, arguments):
    return with_retry(lambda: _exa_once(name, arguments), attempts=6, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa) for blogs, survey pages, project pages, docs and news. Describe the ideal page in natural
    language (e.g. "survey article comparing latent world models"); `objective` says what facts to pull out.
    Returns clean text of the top results with their URLs; sources found here have source family "web"."""
    if not query.strip():
        return "NO RESULTS"
    arguments = {"query": query.strip(), "objective": objective.strip() or f"Find authoritative pages about: {query}",
                 "numResults": _clamp(num_results, 1, 10)}
    try:
        text = _exa_call("web_search_exa", arguments)
        return text[:PAGE_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page, a blog post) as markdown, to get details
    such as numbers, benchmarks or method names. Long pages are truncated to about 12000 characters."""
    if not url.strip().startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()], "maxCharacters": PAGE_CHARS})
        return text[:PAGE_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
