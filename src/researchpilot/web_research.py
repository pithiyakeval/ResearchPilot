"""Bounded web retrieval through the Parallel Search MCP client.

Runs before the research graph, only when web mode is requested: one search, then a few
fetches of validated URLs. Results become extra numbered sources whose text is untrusted
evidence. A failed call never produces placeholder evidence.

Parallel's servers perform the fetches, so this machine never connects to result URLs.
URL checks keep unsafe or internal destinations out of fetch requests and citations.
"""

import asyncio
import ipaddress
import json
import re
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from mcp.types import CallToolResult

from researchpilot.parallel_mcp import ParallelMCPError, ParallelSearchMCPClient
from researchpilot.schemas import WebSource

MAX_SEARCH_RESULTS = 5
MAX_FETCHES = 2
MAX_EXCERPT_CHARS = 1_200
MAX_PAGE_CHARS = 3_000
MAX_TOTAL_CHARS = 12_000
MAX_TITLE_CHARS = 200
MAX_URL_CHARS = 2_048
MAX_QUERY_WORDS = 12
MAX_FETCH_OBJECTIVE_CHARS = 200
RETRIEVAL_DEADLINE_SECONDS = 120.0
# Below this many characters of remaining budget, a source is dropped rather than clipped.
MIN_USEFUL_CHARS = 200
# A fetched page with less meaningful text than this is a low-quality fetch and is skipped.
MIN_PAGE_MEANINGFUL_CHARS = 300

# Whole lines that carry no evidence, compared after lowercasing and trimming punctuation.
_BOILERPLATE_LINES = frozenset(
    {
        "accept",
        "accept all",
        "accept all cookies",
        "accept cookies",
        "back to top",
        "close",
        "cookie settings",
        "create account",
        "forgot password",
        "home",
        "log in",
        "login",
        "menu",
        "privacy policy",
        "register",
        "share",
        "sign in",
        "sign up",
        "skip to content",
        "skip to main content",
        "subscribe",
        "terms of service",
        "terms of use",
    }
)
# A line that is only a Markdown link or image, or a bare URL.
_LINK_ONLY_LINE = re.compile(r"!?\[[^\]]*\]\([^)]*\)|<?https?://\S+>?")
_ALPHANUMERIC = re.compile(r"\w")

_DEFAULT_PORTS = {"http": 80, "https": 443}
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".localdomain", ".internal", ".home.arpa", ".lan")
# Numeric host forms that some resolvers read as IP addresses, e.g. "2130706433" or "0x7f.1".
_NUMERIC_LABEL = re.compile(r"0x[0-9a-f]*|[0-9]+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADING_MARK = re.compile(r"^([ \t]*)#", re.MULTILINE)


class WebRetrievalError(RuntimeError):
    """Web search failed or returned nothing usable; the run must not continue."""


@dataclass(frozen=True)
class WebEvidence:
    """Sources retrieved for one question, with any skipped-item warnings."""

    sources: tuple[WebSource, ...]
    warnings: tuple[str, ...]
    search_calls: int
    fetch_calls: int

    def summary(self) -> str:
        """One-line description of retrieval counts; contains no retrieved text."""
        excerpts = sum(source.kind == "search_excerpt" for source in self.sources)
        pages = len(self.sources) - excerpts
        return (
            f"Web: {self.search_calls} search call(s), {self.fetch_calls} fetch call(s); "
            f"{len(self.sources)} source(s) added ({excerpts} search excerpt(s), "
            f"{pages} fetched page(s)); {len(self.warnings)} warning(s)"
        )


def _is_public_host(host: str) -> bool:
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if len(labels) < 2 or any(not label for label in labels):
            return False
        return not all(_NUMERIC_LABEL.fullmatch(label) for label in labels)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


def normalize_url(url: object) -> str | None:
    """Return a canonical public http(s) URL, or None if it is malformed or unsafe.

    Lowercases the scheme and host, converts international hosts to ASCII, drops the
    fragment, any default port and ``utm_*`` tracking parameters, and rejects credentials,
    local or internal hosts, and IP addresses that are not globally routable. Equal results
    mean the same page.
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or len(url) > MAX_URL_CHARS or any(char.isspace() for char in url):
        return None
    if _CONTROL_CHARS.search(url):
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or "@" in parts.netloc or not parts.hostname:
        return None
    host = parts.hostname.rstrip(".")
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if not _is_public_host(host):
        return None
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", _strip_tracking(parts.query), ""))


def _strip_tracking(query: str) -> str:
    """Drop ``utm_*`` parameters (any case); other parameters keep their order and encoding."""
    kept = [
        part
        for part in query.split("&")
        if part and not unquote(part.split("=", 1)[0]).lower().startswith("utm_")
    ]
    return "&".join(kept)


def meaningful_length(text: str) -> int:
    """Characters of evidence in ``text``, ignoring whitespace runs and boilerplate lines.

    Dropped lines: those with no letters or digits, those that are only a link, image or
    URL, and common navigation or consent phrases such as "Log in" or "Accept all cookies".
    """
    kept: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not _ALPHANUMERIC.search(line) or _LINK_ONLY_LINE.fullmatch(line):
            continue
        if line.lower().strip(" .,:;!?|-*#>_()[]") in _BOILERPLATE_LINES:
            continue
        kept.append(line)
    return len(" ".join(kept))


def _clean_text(text: str, limit: int) -> str:
    """Normalize untrusted text and clip it to ``limit`` characters.

    Leading "#" marks are escaped so retrieved text cannot look like a source heading.
    """
    text = _CONTROL_CHARS.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _HEADING_MARK.sub(r"\1\\#", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " [...]"
    return text


def _clean_title(title: object, url: str) -> str:
    words = " ".join(title.split()) if isinstance(title, str) else ""
    words = _CONTROL_CHARS.sub("", words)[:MAX_TITLE_CHARS].strip()
    return words or (urlsplit(url).hostname or url)


def _payload(result: CallToolResult, tool: str) -> dict[str, Any]:
    """Return the tool's JSON object with a ``results`` list, from structured or text content."""
    structured = result.structured_content
    if isinstance(structured, dict) and isinstance(structured.get("results"), list):
        return structured
    for block in result.content:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return data
    raise WebRetrievalError(f"Parallel {tool} returned results in an unrecognized format.")


def _excerpt_text(item: dict[str, Any]) -> str:
    excerpts = item.get("excerpts")
    if isinstance(excerpts, list):
        parts = [part.strip() for part in excerpts if isinstance(part, str) and part.strip()]
        if parts:
            return "\n\n".join(parts)
    full_content = item.get("full_content")
    return full_content.strip() if isinstance(full_content, str) else ""


def _fetched_page(result: CallToolResult, requested_url: str) -> tuple[object, str]:
    """Return (title, text) for the requested URL, or raise if provenance is not valid.

    Content is accepted only from a result whose reported URL normalizes to the URL that
    was requested; a different or unsafe reported URL (for example a redirect) is rejected.
    """
    items = [item for item in _payload(result, "web_fetch")["results"] if isinstance(item, dict)]
    reported = [normalize_url(item.get("url")) for item in items]
    for item, url in zip(items, reported, strict=True):
        if url == requested_url:
            text = _excerpt_text(item)
            if not text:
                raise WebRetrievalError("the page result had no text")
            return item.get("title"), text
    if any(url is None for url in reported):
        raise WebRetrievalError("the result reported a missing or unsafe URL")
    if reported:
        raise WebRetrievalError("the result reported a different URL (possible redirect)")
    raise WebRetrievalError("the result contained no page")


class _Collector:
    """Adds sources in order while enforcing the per-source and total character budgets."""

    def __init__(self, total_chars: int) -> None:
        self.sources: list[WebSource] = []
        self.warnings: list[str] = []
        self._remaining = total_chars

    def add(self, kind: str, title: object, url: str, text: str, limit: int) -> None:
        if self._remaining < MIN_USEFUL_CHARS:
            self.warnings.append(
                f"Dropped {kind.replace('_', ' ')} for {url}: web text budget used."
            )
            return
        cleaned = _clean_text(text, min(limit, self._remaining))
        if not cleaned:
            self.warnings.append(f"Dropped {kind.replace('_', ' ')} for {url}: no usable text.")
            return
        self._remaining -= len(cleaned)
        self.sources.append(
            WebSource(kind=kind, title=_clean_title(title, url), url=url, text=cleaned)
        )


def search_query_for(question: str) -> str:
    """Keyword query for the search tool: the first words of the question."""
    return " ".join(question.split()[:MAX_QUERY_WORDS])


def fetch_order(excerpts: list[str]) -> list[int]:
    """Indexes of search results in the order to fetch them: most substantive excerpt first.

    Excerpt size is measured like page text and capped at the excerpt clip length, so long
    excerpts tie; ties keep the search rank. The result is deterministic.
    """
    sizes = [min(meaningful_length(text), MAX_EXCERPT_CHARS) for text in excerpts]
    return sorted(range(len(excerpts)), key=lambda index: (-sizes[index], index))


async def gather_web_evidence(
    question: str,
    *,
    client_factory: Callable[[], ParallelSearchMCPClient] | None = None,
    max_search_results: int = MAX_SEARCH_RESULTS,
    max_fetches: int = MAX_FETCHES,
    total_chars: int = MAX_TOTAL_CHARS,
    deadline_seconds: float = RETRIEVAL_DEADLINE_SECONDS,
) -> WebEvidence:
    """Search once, fetch up to ``max_fetches`` result pages, and return bounded evidence.

    Pages are fetched for the results with the most substantive excerpts (see
    ``fetch_order``). Raises ``WebRetrievalError`` if connecting or searching fails, times
    out, or yields no usable result. A failed or low-quality fetch is skipped with a warning
    and the search excerpts are kept.
    """
    factory = client_factory or ParallelSearchMCPClient
    query = search_query_for(question)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_seconds
    collector = _Collector(total_chars)
    fetch_calls = 0

    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout_at(deadline):
                client = await stack.enter_async_context(factory())
                search_result = await client.search(question, [query])
        except TimeoutError as exc:
            raise WebRetrievalError(
                f"Web search did not finish within {deadline_seconds:g} seconds."
            ) from exc
        except ParallelMCPError as exc:
            raise WebRetrievalError(f"Web search failed: {exc}") from exc

        candidates: list[tuple[str, object, str]] = []
        seen: set[str] = set()
        for item in _payload(search_result, "web_search")["results"]:
            if len(candidates) >= max_search_results:
                break
            if not isinstance(item, dict):
                collector.warnings.append("Skipped a malformed search result.")
                continue
            url = normalize_url(item.get("url"))
            if url is None:
                collector.warnings.append("Skipped a search result with a missing or unsafe URL.")
                continue
            if url in seen:
                continue
            seen.add(url)
            text = _excerpt_text(item)
            if not text:
                collector.warnings.append(f"Skipped search result {url}: no excerpt text.")
                continue
            candidates.append((url, item.get("title"), text))
        if not candidates:
            raise WebRetrievalError("Web search returned no usable results.")

        for url, title, text in candidates:
            collector.add("search_excerpt", title, url, text, MAX_EXCERPT_CHARS)

        order = fetch_order([text for _, _, text in candidates])
        for url, search_title, _ in (candidates[index] for index in order[:max_fetches]):
            if loop.time() >= deadline:
                collector.warnings.append(f"Skipped fetch of {url}: retrieval deadline reached.")
                continue
            fetch_calls += 1
            try:
                async with asyncio.timeout_at(deadline):
                    result = await client.fetch(
                        url,
                        objective=question[:MAX_FETCH_OBJECTIVE_CHARS],
                        search_queries=[query],
                    )
                page_title, page_text = _fetched_page(result, url)
            except TimeoutError:
                collector.warnings.append(f"Skipped fetch of {url}: retrieval deadline reached.")
                continue
            except (ParallelMCPError, WebRetrievalError) as exc:
                collector.warnings.append(f"Skipped fetch of {url}: {exc}")
                continue
            size = meaningful_length(page_text)
            if size < MIN_PAGE_MEANINGFUL_CHARS:
                collector.warnings.append(
                    f"Skipped fetch of {url}: low-quality page with {size} meaningful "
                    f"character(s), below the minimum of {MIN_PAGE_MEANINGFUL_CHARS}."
                )
                continue
            collector.add(
                "fetched_page", page_title or search_title, url, page_text, MAX_PAGE_CHARS
            )

    return WebEvidence(
        sources=tuple(collector.sources),
        warnings=tuple(collector.warnings),
        search_calls=1,
        fetch_calls=fetch_calls,
    )
