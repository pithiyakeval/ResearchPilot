"""Opt-in evidence capture: the exact sources and analysis notes one run used, as JSON.

Written only when ``--save-sources`` is given. The file holds full source text and analysis
notes, which may be private, so nothing from it is printed or logged. It never holds full
prompts, raw evaluator replies, environment variables or the Ollama base URL.
"""

import contextlib
import hashlib
import json
import ntpath
import os
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchpilot.config import Settings
from researchpilot.prompts import neutralize_delimiters
from researchpilot.schemas import ResearchReport, ResearchRequest
from researchpilot.state import ResearchState

if TYPE_CHECKING:
    from researchpilot.web_research import WebEvidence

# Version 2: sources carry numbered_title (and page_title for web sources) instead of title;
# text_handling is an object; source_block_format and output_file_sha256 were added.
SCHEMA_VERSION = 2
# How the per-source texts rebuild the numbered source block, as a Python expression.
SOURCE_BLOCK_FORMAT = '"\\n\\n".join(f"[{id}]\\n{text}")'
# Each statement is checked against the code by tests/test_capture.py.
TEXT_HANDLING = {
    "prompts": (
        "Every analysis, evaluation and report prompt contained the source block "
        "(source_block_format applied to sources, in order) after neutralize_delimiters: a "
        "'<' that opens or closes a prompt delimiter tag (source_material, analysis_notes, "
        "previous_analysis_notes, coverage_gaps, run_status; any case) becomes '&lt;'. "
        "source_block_sha256 is the SHA-256 of that escaped block in UTF-8. Analysis notes and "
        "gap lists were escaped the same way wherever a prompt included them. The texts stored "
        "here are not escaped."
    ),
    "local_sources": (
        "The supplied file is split at top-level Markdown headings ('# Title') outside fenced "
        "code blocks. Before splitting, label-like text such as [S2] or [ s 2 ] is rewritten "
        "as (S2) everywhere, headings included. Blank lines at the start and end of each source "
        "are dropped. text is the heading line and content as numbered; numbered_title is the "
        "heading text."
    ),
    "web_sources": (
        "text is a generated line '# ' + numbered_title, then the excerpt or page text with "
        "label-like text rewritten as for local sources and outer whitespace trimmed. "
        "numbered_title is '<evidence type>: <page title> — <url>' with label-like text "
        "rewritten and whitespace collapsed. When retrieved with --web, the excerpt or page "
        "text was cleaned before numbering: line endings normalized to \\n, control characters "
        "removed, a '#' starting a line (after optional spaces or tabs) escaped as '\\#', runs "
        "of three or more newlines collapsed to two, and text over its length limit clipped "
        "and marked with ' [...]'. page_title is the page title after retrieval cleanup "
        "(whitespace collapsed, control characters removed, at most 200 characters, the URL's "
        "host name when the page had no title) but before label rewriting."
    ),
    "analysis_notes": (
        "Each pass's notes are the model's reply with surrounding whitespace trimmed, as later "
        "prompts received them before escaping."
    ),
    "evaluation_gaps": (
        "analysis_gaps, source_gaps and conflicts are the evaluator's validated lists: each "
        "item trimmed of surrounding whitespace and 1-300 characters long, at most 8 items per "
        "list. An invalid reply is stored as status 'invalid' with its validation error and "
        "empty lists; the raw reply is not stored."
    ),
}


class CaptureError(Exception):
    """The evidence file cannot be built or saved; the message is safe to show the user."""


# ntpath.isreserved exists from Python 3.13; older versions use the device-name list below.
_ISRESERVED = getattr(ntpath, "isreserved", None)
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{digit}" for port in ("COM", "LPT") for digit in "123456789¹²³"}
)
# Characters Windows rejects in file names (':' is reported separately as a data stream).
_FORBIDDEN_NAME_CHARS = frozenset('<>"/\\|?*')


def strip_extended_prefix(path: str) -> str:
    r"""Remove a Windows extended-length prefix: ``\\?\C:\x`` -> ``C:\x``, ``\\?\UNC\s\x`` -> ``\\s\x``."""
    if not path.startswith("\\\\?\\"):
        return path
    rest = path[4:]
    if rest[:4].upper() == "UNC\\":
        return "\\\\" + rest[4:]
    return rest


def _is_reserved_name(name: str) -> bool:
    if _ISRESERVED is not None:
        return _ISRESERVED(name)
    # Windows ignores the extension and trailing dots or spaces when matching device names.
    return name.rstrip(" .").split(".", 1)[0].rstrip(" ").upper() in _RESERVED_DEVICE_NAMES


def windows_path_problem(path: str) -> str | None:
    r"""Why ``path`` is unsafe as a Windows file destination, or None if it is not.

    Uses Windows path rules on every platform, so it can be tested anywhere. Rejects device
    paths (``\\.\...``), a ``:`` after the drive (an alternate data stream, which would write
    into another file), names ending in a dot or space (Windows would save a different
    name), characters Windows forbids in names, and reserved device names such as NUL, CON
    or COM1.
    """
    text = strip_extended_prefix(path)
    if text.startswith(("\\\\.\\", "//./")):
        return "is a Windows device path"
    _drive, rest = ntpath.splitdrive(text)
    if ":" in rest:
        return "contains ':', which Windows treats as an alternate data stream of another file"
    name = ntpath.basename(rest)
    if name != name.rstrip(" ."):
        return "ends with a dot or space, which Windows silently removes from file names"
    if any(char in _FORBIDDEN_NAME_CHARS or ord(char) < 32 for char in name):
        return "contains a character Windows does not allow in file names"
    if _is_reserved_name(name):
        return "uses a reserved Windows device name"
    return None


def _path_key(path: Path) -> str:
    """Comparable form of a path: no extended-length prefix, absolute, links resolved, and
    case-folded on Windows."""
    plain = strip_extended_prefix(os.fspath(path))
    return os.path.normcase(os.path.realpath(os.path.abspath(plain)))


def validate_capture_path(
    path: Path, *, source_file: Path | None = None, report_output: Path | None = None
) -> Path:
    r"""Check an evidence-file destination before the run; return it as an absolute path.

    The name must end in ``.json``; on Windows it must not be a device path, an alternate
    data stream or a reserved device name; nothing may exist at the path (file, folder or
    link); the parent folder must already exist (it is never created); and the path must
    differ from the source file and the report output path, including through an
    extended-length (``\\?\``) alias. Nothing is written here.
    """
    absolute = Path(os.path.abspath(path))
    if absolute.suffix.lower() != ".json":
        raise CaptureError(f"{absolute} must end in .json.")
    if os.name == "nt":
        # Check the path as typed too: abspath already drops a trailing dot or space.
        for form in (os.fspath(path), str(absolute)):
            problem = windows_path_problem(form)
            if problem is not None:
                raise CaptureError(f"{form!r} {problem}; choose a different name.")
    key = _path_key(absolute)
    if source_file is not None and key == _path_key(source_file):
        raise CaptureError(f"{absolute} is the source file; choose a different path.")
    if report_output is not None and key == _path_key(report_output):
        raise CaptureError(f"{absolute} is the report output path; choose a different path.")
    if os.path.lexists(absolute):
        raise CaptureError(f"{absolute} already exists; refusing to overwrite it.")
    parent = absolute.parent
    if not parent.exists():
        raise CaptureError(f"The folder {parent} does not exist; create it first.")
    if not parent.is_dir():
        raise CaptureError(f"{parent} is not a folder.")
    return absolute


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create ``path`` only if it does not exist; remove it again if writing fails."""
    with open(path, "xb") as handle:
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
            raise


def write_capture(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as UTF-8 JSON to a new file at ``path``, atomically and never overwriting.

    The JSON is serialized before anything touches the disk, written to a temporary file in
    the same folder, flushed to disk, and then published with a hard link, which fails
    instead of replacing an existing file. Where hard links are unavailable, an exclusive
    create is used instead. The temporary file is always removed; a failure never leaves a
    partial evidence file behind.
    """
    try:
        payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"The evidence data could not be serialized ({exc}).") from exc
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except OSError:
            # Filesystems without hard links: create the file exclusively instead.
            _write_exclusive(path, payload)
    except FileExistsError as exc:
        raise CaptureError(f"{path} already exists; refusing to overwrite it.") from exc
    except OSError as exc:
        raise CaptureError(f"Could not save evidence file {path}: {exc.strerror or exc}") from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of the exact bytes in ``path``, such as a report file read back after writing."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CaptureError(
            f"Could not read back the report file {path}: {exc.strerror or exc}"
        ) from exc


def _tool_version() -> str:
    try:
        return version("researchpilot")
    except PackageNotFoundError:
        return "unknown"


def _sources(request: ResearchRequest, final: ResearchState) -> list[dict[str, Any]]:
    web_ids = final["web_source_ids"]
    if len(web_ids) != len(request.web_sources):
        raise CaptureError("Captured web sources do not match the sources numbered for the run.")
    web_by_id = dict(zip(web_ids, request.web_sources, strict=True))
    sources: list[dict[str, Any]] = []
    for source_id, numbered_title in final["source_titles"].items():
        web = web_by_id.get(source_id)
        if web is None:
            entry: dict[str, Any] = {
                "id": source_id,
                "type": "local",
                "numbered_title": numbered_title,
            }
        else:
            entry = {
                "id": source_id,
                "type": "web",
                "evidence_type": web.kind,
                "numbered_title": numbered_title,
                "page_title": web.title,
                "url": web.url,
            }
        entry["text"] = final["source_texts"][source_id]
        sources.append(entry)
    return sources


def build_capture(
    request: ResearchRequest,
    final: ResearchState,
    report: ResearchReport,
    *,
    settings: Settings,
    retrieval: "WebEvidence | None",
    output_path: str | None,
    now: datetime,
    output_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the version-2 evidence record for one completed run.

    ``output_file_sha256`` is the hash of the report file's bytes as written (see
    ``file_sha256``), or None when no report file was written.

    Raises ``CaptureError`` if the per-source texts do not rebuild the exact numbered block
    the prompts received, so a file that cannot reproduce the run is never written.
    """
    sources = _sources(request, final)
    rebuilt = "\n\n".join(f"[{source['id']}]\n{source['text']}" for source in sources)
    if rebuilt != final["numbered_sources"]:
        raise CaptureError("Captured source texts do not match the block the model received.")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "researchpilot", "version": _tool_version()},
        "captured_at": now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "question": request.question,
        "source_mode": "local+web" if request.web_sources else "local",
        # Only non-secret model settings; the base URL could carry credentials.
        "settings": {
            "ollama_model": settings.ollama_model,
            "ollama_temperature": settings.ollama_temperature,
            "ollama_num_ctx": settings.ollama_num_ctx,
            "ollama_num_predict": settings.ollama_num_predict,
            "max_analysis_passes": settings.researchpilot_max_analysis_passes,
        },
        "text_handling": dict(TEXT_HANDLING),
        "source_block_format": SOURCE_BLOCK_FORMAT,
        "source_block_sha256": _sha256(neutralize_delimiters(final["numbered_sources"])),
        "sources": sources,
        "retrieval": None
        if retrieval is None
        else {
            "search_calls": retrieval.search_calls,
            "fetch_calls": retrieval.fetch_calls,
            "warnings": list(retrieval.warnings),
        },
        "analysis_passes": [
            {"pass": record["pass_number"], "notes": record["notes"]}
            for record in final["analysis_history"]
        ],
        "evaluation_history": [
            {
                "pass": record["pass_number"],
                "status": record["status"],
                "analysis_gaps": record["analysis_gaps"],
                "source_gaps": record["source_gaps"],
                "conflicts": record["conflicts"],
                "error": record["error"],
            }
            for record in final["evaluation_history"]
        ],
        "stop_reason": final["stop_reason"],
        "analysis_pass_count": final["pass_count"],
        "unresolved_analysis_gaps": report.unresolved_analysis_gaps,
        "report_metadata": {
            "output_path": output_path,
            # SHA-256 of the --output file's exact bytes as written: on Windows the CLI writes
            # CRLF line endings and a final newline, so this differs from markdown_sha256.
            "output_file_sha256": output_file_sha256,
            # SHA-256 of the report Markdown string itself (UTF-8, LF, no added newline).
            "markdown_sha256": _sha256(report.markdown),
            "markdown_chars": len(report.markdown),
            "invalid_citations": report.invalid_citations,
            "quality_warnings": report.quality_warnings,
            "model_calls": report.model_calls,
            "elapsed_seconds": report.elapsed_seconds,
            "node_seconds": report.node_seconds,
        },
    }
