"""Serve the API with uvicorn: ``uv run python -m researchpilot.api``.

A separate entry point, because adding a second command to the Typer CLI would change how
the existing ``researchpilot "question" --source-file ...`` invocation is parsed.
"""

import uvicorn

from researchpilot.api.app import create_app
from researchpilot.api.settings import ApiSettings


def main() -> None:
    api_settings = ApiSettings()
    uvicorn.run(
        create_app(api_settings=api_settings), host=api_settings.host, port=api_settings.port
    )


if __name__ == "__main__":
    main()
