"""Command-line interface for ResearchPilot."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from researchpilot.capture import (
    CaptureError,
    build_capture,
    file_sha256,
    validate_capture_path,
    write_capture,
)
from researchpilot.config import get_settings
from researchpilot.errors import InputError, InputTooLargeError, ModelRunError, RetrievalError

# Re-exported: callers and tests use cli.build_model, and main() looks it up here at run time.
from researchpilot.models import build_model
from researchpilot.pipeline import prepare_request, run_research
from researchpilot.schemas import ResearchReport

if TYPE_CHECKING:
    # Type-only: offline runs never import the MCP client.
    from researchpilot.web_research import WebEvidence

__all__ = ["app", "build_model", "main", "run_summary"]

app = typer.Typer(no_args_is_help=True, help="Synthesize supplied research sources.")
console = Console()
err_console = Console(stderr=True)


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="Research question to answer.")],
    source_file: Annotated[
        Path,
        typer.Option(
            "--source-file",
            "-s",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="UTF-8 Markdown or text file containing source notes/excerpts.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to save the Markdown report."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Log timing for each model call to stderr."),
    ] = False,
    web: Annotated[
        bool,
        typer.Option(
            "--web",
            "-w",
            help=(
                "Also search the web with Parallel Search MCP (one search, a few page "
                "fetches) and add the results as untrusted sources."
            ),
        ),
    ] = False,
    save_sources: Annotated[
        Path | None,
        typer.Option(
            "--save-sources",
            help=(
                "Save the exact source text, analysis notes and evaluation history used in "
                "this run to a new .json file. Refuses to overwrite and does not create "
                "folders. The file contains full source text, which may be private."
            ),
        ),
    ] = None,
) -> None:
    """Generate a report from the supplied source material, plus web results with --web."""
    if verbose:
        _enable_timing_logs()
    try:
        # Checked before any model or network call, so a bad path wastes no work.
        capture_path = (
            None
            if save_sources is None
            else validate_capture_path(save_sources, source_file=source_file, report_output=output)
        )
        source_material = source_file.read_text(encoding="utf-8")
        request = prepare_request(question, source_material)
        settings = get_settings()
        # build_model is looked up in this module at call time, so it can be replaced in tests.
        outcome = run_research(
            request,
            settings=settings,
            web=web,
            model_factory=build_model,
            on_retrieval=_show_retrieval,
        )
    except typer.Exit:
        raise
    except CaptureError as exc:
        console.print(f"[red]Cannot save evidence file:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except (InputError, ValidationError, UnicodeDecodeError) as exc:
        console.print(f"[red]Input validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except InputTooLargeError as exc:
        console.print(f"[red]Input too large for the model context:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except RetrievalError as exc:
        console.print(f"[red]Web retrieval failed:[/red] {exc} No model was called.")
        raise typer.Exit(code=1) from exc
    except ModelRunError as exc:
        # Kept from before the pipeline existed: any OSError during the run (including the
        # ConnectionError the Ollama client raises when the server is down) has always been
        # reported this way.
        if isinstance(exc.__cause__, OSError):
            console.print(f"[red]Could not read input file:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        console.print(f"[red]Research run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]Could not read input file:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]Research run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    request, report, final_state = outcome.request, outcome.report, outcome.final_state
    retrieval = outcome.retrieval

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.markdown + "\n", encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Could not save report:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]Report saved to {output}[/green]")

    console.print(Markdown(report.markdown))
    err_console.print(run_summary(report), style="dim")

    if capture_path is not None:
        try:
            data = build_capture(
                request,
                final_state,
                report,
                settings=settings,
                retrieval=retrieval,
                output_path=None if output is None else str(output),
                # Read back, so the hash covers the platform's line endings as written.
                output_file_sha256=None if output is None else file_sha256(output),
                now=datetime.now(UTC),
            )
            write_capture(capture_path, data)
        except CaptureError as exc:
            # The report is already printed and saved; the requested evidence file is not.
            console.print(f"[red]Evidence file not saved:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        err_console.print(
            f"Evidence saved to {capture_path} ({len(data['sources'])} source(s), "
            f"{len(data['analysis_passes'])} analysis pass(es)). It contains full source text "
            "and analysis notes.",
            style="dim",
        )


def run_summary(report: ResearchReport) -> str:
    """One-line description of passes, model calls and timing; contains no source text."""
    node_seconds = report.node_seconds or {}
    nodes = ", ".join(f"{node} {seconds:.1f}s" for node, seconds in node_seconds.items())
    summary = (
        f"Run: {report.analysis_passes} analysis pass(es), {report.model_calls} model call(s), "
        f"{report.elapsed_seconds or 0:.1f}s total ({nodes}); stop reason: {report.stop_reason}"
    )
    if report.unresolved_analysis_gaps:
        summary += f"; {len(report.unresolved_analysis_gaps)} unresolved analysis gap(s)"
    return summary


def _show_retrieval(evidence: "WebEvidence") -> None:
    """Print web retrieval warnings and counts to stderr, before any model is called."""
    for warning in evidence.warnings:
        err_console.print(f"Web warning: {warning}", style="yellow")
    err_console.print(evidence.summary(), style="dim")


def _enable_timing_logs() -> None:
    package_logger = logging.getLogger("researchpilot")
    package_logger.setLevel(logging.INFO)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        package_logger.addHandler(handler)


if __name__ == "__main__":
    app()
