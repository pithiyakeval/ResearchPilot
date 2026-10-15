"""HTTP API for ResearchPilot. Run it with ``uv run python -m researchpilot.api``."""

from researchpilot.api.app import create_app

__all__ = ["create_app"]
