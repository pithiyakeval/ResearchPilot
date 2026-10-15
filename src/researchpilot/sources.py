"""Deterministic source numbering and citation-reference checks.

Supplied material is split at top-level Markdown headings (``# Title``) and each block is
given a stable ID: [S1], [S2], ... in order of appearance. Reports must cite those IDs, and
the citations are checked in code after generation.

A valid ID only shows which supplied source a sentence points to. It does not show that the
source supports the claim.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

UNTITLED_SOURCE = "Untitled source"
PREAMBLE_TITLE = "Untitled text before the first heading"

# A top-level ATX heading: "# Title", optionally closed with "#"s; "##" and deeper stay inside.
_HEADING = re.compile(r"^ {0,3}#[ \t]+(?P<title>.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
# Anything in the supplied text that looks like a source label, e.g. "[S2]" or "[ s 2 ]".
_LABEL_LIKE = re.compile(r"\[\s*([Ss])\s*(\d+)\s*\]")
# Citations in model output: "[S1]", "[S1, S3]" or "[S1; S2]".
_CITATION_GROUP = re.compile(r"\[\s*([Ss]\d+(?:\s*[,;]\s*[Ss]\d+)*)\s*\]")
_INTERNAL_REFERENCE = re.compile(r"\b(analysis notes?|run status)\b", re.IGNORECASE)

# Names of internal prompt blocks; every prompt delimiter tag must be listed (a test checks).
INTERNAL_LABELS = (
    "previous_analysis_notes",
    "analysis_notes",
    "source_material",
    "coverage_gaps",
    "run_status",
)
_LABEL_NAMES = "|".join(INTERNAL_LABELS)
# An internal label written like a citation or a tag: inside [...] (not a Markdown link
# "[text](url)"), <...>, </...> or {...}. Bare names in prose or `code` are not flagged, since
# a report may legitimately discuss them.
_INTERNAL_LABEL_FORM = re.compile(
    rf"\[[^\[\]\n]*?\b(?:{_LABEL_NAMES})\b[^\[\]\n]*?\](?!\()"
    rf"|</?\s*(?:{_LABEL_NAMES})\s*>"
    rf"|\{{\s*(?:{_LABEL_NAMES})\s*\}}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumberedSources:
    text: str
    """Source material with an ID line before each block, ready for prompts."""
    titles: dict[str, str]
    """Source ID (``"S1"``) to title, in order."""
    extra_ids: tuple[str, ...] = ()
    """IDs given to ``extra_sources``, which always follow the supplied material's IDs."""
    texts: dict[str, str] = field(default_factory=dict)
    """Source ID to that source's text (title line and content), as the prompts show it."""


def _defuse_labels(text: str) -> str:
    """Rewrite label-like text such as "[S2]" as "(S2)" so a source cannot forge an ID."""
    return _LABEL_LIKE.sub(lambda match: f"({match.group(1).upper()}{match.group(2)})", text)


def _strip_blank_edges(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def number_sources(
    source_material: str, extra_sources: Sequence[tuple[str, str]] = ()
) -> NumberedSources:
    """Split material at top-level headings and label each block [S1], [S2], ...

    Text before the first heading becomes its own source; material without headings is a
    single source. Headings inside fenced code blocks do not start a source. Apart from
    defusing label-like text, titles and content are kept as written.

    ``extra_sources`` are (title, text) pairs, such as retrieved web evidence, numbered after
    the supplied material in order. Each is exactly one source: its text is never split, so
    it cannot create, merge or hide sources, and it gets the same label defusing.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    in_fence = False
    for line in _defuse_labels(source_material).splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            blocks.append((current_title or PREAMBLE_TITLE, current_lines))
            current_title, current_lines = heading.group("title").strip(), [line]
        else:
            current_lines.append(line)
    if current_title is None:
        blocks.append((UNTITLED_SOURCE, current_lines))
    else:
        blocks.append((current_title, current_lines))

    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    rendered: list[str] = []
    for title, lines in blocks:
        content = _strip_blank_edges(lines)
        if not content:
            continue
        source_id = f"S{len(titles) + 1}"
        titles[source_id] = title
        texts[source_id] = "\n".join(content)
        rendered.append(f"[{source_id}]\n" + texts[source_id])

    extra_ids: list[str] = []
    for title, text in extra_sources:
        clean_title = " ".join(_defuse_labels(title).split()) or UNTITLED_SOURCE
        source_id = f"S{len(titles) + 1}"
        titles[source_id] = clean_title
        extra_ids.append(source_id)
        texts[source_id] = f"# {clean_title}\n{_defuse_labels(text).strip()}"
        rendered.append(f"[{source_id}]\n" + texts[source_id])
    return NumberedSources(
        text="\n\n".join(rendered), titles=titles, extra_ids=tuple(extra_ids), texts=texts
    )


def cited_source_ids(report: str, titles: dict[str, str]) -> list[str]:
    """Supplied source IDs the report cites, in first-cited order and without duplicates.

    Uses the same citation pattern as ``check_citations``; IDs that are not supplied sources
    are left out (``check_citations`` reports those).
    """
    cited: list[str] = []
    for group in _CITATION_GROUP.findall(report):
        cited.extend(part.strip().upper() for part in re.split(r"[,;]", group))
    return [sid for sid in dict.fromkeys(cited) if sid in titles]


def check_citations(report: str, titles: dict[str, str]) -> list[str]:
    """Return problems with the report's source references; an empty list means none found.

    Flags unknown IDs, internal labels written like citations or tags (``[run_status]``),
    references to internal pipeline steps (analysis notes, run status), and a report with no
    valid source ID at all. The report text is never changed.
    """
    problems: list[str] = []
    cited: list[str] = []
    for group in _CITATION_GROUP.findall(report):
        cited.extend(part.strip().upper() for part in re.split(r"[,;]", group))

    unknown = sorted(set(cited) - set(titles), key=lambda sid: int(sid[1:]))
    problems.extend(f"[{sid}] is not a supplied source ID" for sid in unknown)

    labels = dict.fromkeys(match.group(0) for match in _INTERNAL_LABEL_FORM.finditer(report))
    problems.extend(f'"{label}" looks like an internal label, not a source' for label in labels)

    internal = sorted({match.lower() for match in _INTERNAL_REFERENCE.findall(report)})
    problems.extend(
        f'"{phrase}" is an internal processing step, not a source' for phrase in internal
    )

    if titles and not any(sid in titles for sid in cited):
        problems.append("the report cites no supplied source ID")
    return problems


def citation_warning(problems: list[str], titles: dict[str, str]) -> str:
    """Deterministic note appended to a report whose source references failed the check."""
    key = "; ".join(f"[{sid}] {title}" for sid, title in titles.items())
    return (
        f"> **ResearchPilot citation check:** {'; '.join(problems)}. The report text above "
        f"is unchanged. Supplied source IDs: {key}. A valid ID shows which supplied source "
        "is referenced; it does not confirm that the source supports the claim."
    )


def web_source_note(source_ids: Sequence[str], titles: dict[str, str]) -> str:
    """Deterministic note listing each web source ID with its evidence type, title and URL."""
    key = "; ".join(f"[{sid}] {titles[sid]}" for sid in source_ids)
    return (
        "> **ResearchPilot web sources:** retrieved through Parallel Search MCP before "
        f"analysis and treated as untrusted evidence: {key}. A valid ID shows which retrieved "
        "text is referenced; it does not confirm that the page supports the claim."
    )
