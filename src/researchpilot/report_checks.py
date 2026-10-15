"""Heuristic review hints for generated reports.

These are prompts for a human reviewer, not findings. They never change the report text,
the citation check, or the exit code. Both detectors are word-pattern checks: they will miss
problems worded differently and sometimes flag a correct sentence, and a report without
hints is not thereby supported by its sources.
"""

import re

# Words that turn a claim about one source into a general claim.
GENERALIZING_WORDS = (
    "often",
    "frequently",
    "commonly",
    "generally",
    "typically",
    "usually",
    "widely",
)
# A claim that the sources as a whole lack something, e.g. "the sources do not address X".
# Statements that one source reports something missing ("[S1] did not report X") are often
# correct and are deliberately not treated as absence claims.
_CORPUS_ABSENCE = re.compile(
    r"\b(?:the|these|supplied|provided|available)\s+sources?\s+(?:do|does|did)\s+not\b"
    r"|\bno\s+(?:supplied\s+|provided\s+)?sources?\b"
    r"|\bnone\s+of\s+the\s+sources\b",
    re.IGNORECASE,
)
# Two statements must share at least this many content words to be compared.
MIN_SHARED_TERMS = 3
EXCERPT_CHARS = 80

_CITATION_GROUP = re.compile(r"\[\s*([Ss]\d+(?:\s*[,;]\s*[Ss]\d+)*)\s*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\[])")
_BULLET = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+")
_WORD = re.compile(r"[a-z]{4,}")
# Common words, and words that describe sources rather than a topic, are not content terms.
_STOPWORDS = frozenset(
    {
        *("about", "also", "among", "because", "been", "being", "between", "both", "could"),
        *("does", "from", "have", "into", "many", "more", "most", "much", "only", "other"),
        *("over", "same", "some", "such", "than", "that", "their", "them", "then", "there"),
        *("these", "they", "this", "those", "through", "under", "very", "were", "what"),
        *("when", "where", "which", "while", "with", "within", "would", "your"),
        *("address", "addressed", "cover", "covered", "discuss", "discussed", "mention"),
        *("mentioned", "provide", "provided", "report", "reported", "source", "sources"),
        *("supplied", "specific", "potential"),
    }
)
_SUFFIXES = ("ing", "es", "ed", "s")


def _sentences(report: str) -> list[str]:
    """Report text as single sentences, with list markers and bold markup removed."""
    sentences: list[str] = []
    for line in report.splitlines():
        line = _BULLET.sub("", line).replace("**", "").strip()
        if line and not line.startswith(("#", ">")):
            sentences.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return sentences


def _cited_ids(sentence: str) -> list[str]:
    ids: list[str] = []
    for group in _CITATION_GROUP.findall(sentence):
        ids.extend(part.strip().upper() for part in re.split(r"[,;]", group))
    return list(dict.fromkeys(ids))


def _excerpt(sentence: str) -> str:
    text = " ".join(_CITATION_GROUP.sub("", sentence).split())
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_terms(sentence: str) -> dict[str, str]:
    """Stem to the first word with that stem, for words of four or more letters."""
    terms: dict[str, str] = {}
    for word in _WORD.findall(_CITATION_GROUP.sub("", sentence).lower()):
        if word not in _STOPWORDS:
            terms.setdefault(_stem(word), word)
    return terms


def find_unscoped_generalizations(report: str, source_texts: dict[str, str]) -> list[str]:
    """Review hints for cited sentences whose generalizing word no cited source uses.

    A review signal, not an evidence-support detector: a source can support a general claim
    in other words, and a source that uses the word may still not support the claim.
    Sentences without citations, or citing only unknown IDs, are left to other checks.
    """
    hints: list[str] = []
    for sentence in _sentences(report):
        cited = [sid for sid in _cited_ids(sentence) if sid in source_texts]
        if not cited:
            continue
        cited_text = " ".join(source_texts[sid] for sid in cited).lower()
        for word in GENERALIZING_WORDS:
            pattern = rf"\b{word}\b"
            if re.search(pattern, sentence, re.IGNORECASE) and not re.search(pattern, cited_text):
                labels = ", ".join(f"[{sid}]" for sid in cited)
                hints.append(
                    f'"{_excerpt(sentence)}" {labels} may generalize beyond its cited '
                    f'source(s): the word "{word}" does not appear in them. Check whether the '
                    "sources support a general claim."
                )
                break
    return hints


def find_possible_contradictions(report: str) -> list[str]:
    """Review hints pairing a claim that the sources lack something with a cited claim on it.

    Only claims about the sources as a whole ("the sources do not address X") are compared,
    against cited statements that are not themselves such claims and share at least
    ``MIN_SHARED_TERMS`` content words. Misses contradictions between positive statements,
    absence claims worded differently, and pairs that use synonyms instead of shared words.
    """
    sentences = _sentences(report)
    absences = [sentence for sentence in sentences if _CORPUS_ABSENCE.search(sentence)]
    claims = [
        sentence
        for sentence in sentences
        if _cited_ids(sentence) and not _CORPUS_ABSENCE.search(sentence)
    ]
    hints: list[str] = []
    for absence in absences:
        absence_terms = _content_terms(absence)
        for claim in claims:
            shared = absence_terms.keys() & _content_terms(claim).keys()
            if len(shared) >= MIN_SHARED_TERMS:
                words = ", ".join(sorted(absence_terms[stem] for stem in shared))
                hints.append(
                    f'Possible contradiction: "{_excerpt(absence)}" vs "{_excerpt(claim)}" '
                    f"(shared terms: {words}). Check both statements against the sources."
                )
    return hints


def review_hints(report: str, source_texts: dict[str, str]) -> list[str]:
    """All heuristic review hints for the model's report text, in a fixed order."""
    return find_unscoped_generalizations(report, source_texts) + find_possible_contradictions(
        report
    )


def review_note(hints: list[str]) -> str:
    """Deterministic note listing review hints and what they do not establish."""
    lines = "\n".join(f"> - {hint}" for hint in hints)
    return (
        "> **ResearchPilot review hints (heuristic):** automated prompts for human review, "
        "not findings. They can be wrong in either direction, and the absence of hints does "
        f"not mean claims are supported.\n>\n{lines}"
    )
