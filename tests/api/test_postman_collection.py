"""Offline checks that the Postman collection and the README match the running app's contract.

The app's OpenAPI schema is the reference, so nothing here copies the schemas. Example bodies
go through the real validation, but no request is sent: no model, database, network or MCP
server is contacted.
"""

import json
import re
from pathlib import Path

import pytest

from researchpilot.api import create_app
from researchpilot.api.job_routes import JobSubmission
from researchpilot.api.settings import ApiSettings
from researchpilot.config import Settings
from researchpilot.errors import InputError
from researchpilot.pipeline import prepare_request

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "postman" / "ResearchPilot.postman_collection.json"
ENVIRONMENT = ROOT / "postman" / "ResearchPilot.local.postman_environment.json"
README = ROOT / "README.md"
SCHEMA_V21 = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
VARIABLE = re.compile(r"\{\{(\w+)\}\}")


def requests(items: list[dict], prefix: str = "") -> list[tuple[str, dict]]:
    found = []
    for item in items:
        name = f"{prefix}{item['name']}"
        if "item" in item:
            found += requests(item["item"], f"{name} / ")
        else:
            found.append((name, item))
    return found


@pytest.fixture(scope="module")
def collection() -> dict:
    return json.loads(COLLECTION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def environment() -> dict:
    return json.loads(ENVIRONMENT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def openapi() -> dict:
    app = create_app(
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
        api_settings=ApiSettings(_env_file=None),  # type: ignore[call-arg]
        ollama_probe=lambda: True,
    )
    return app.openapi()


def operation(item: dict) -> tuple[str, str]:
    """The request's method and its path with ``{{variable}}`` segments as ``{param}``."""
    request = item["request"]
    url = request["url"]
    assert url["host"] == ["{{baseUrl}}"]
    assert url["raw"] == "{{baseUrl}}/" + "/".join(url["path"])
    segments = ["{job_id}" if VARIABLE.fullmatch(part) else part for part in url["path"]]
    return request["method"].lower(), "/" + "/".join(segments)


def test_the_collection_is_a_postman_v21_collection(collection) -> None:
    assert collection["info"]["schema"] == SCHEMA_V21
    assert collection["info"]["name"] == "ResearchPilot API"
    for name, item in requests(collection["item"]):
        assert set(item) <= {"name", "request", "event", "response"}, name
        assert item["request"]["method"] in {"GET", "POST"}, name
        for event in item.get("event", []):
            assert event["listen"] == "test"
            assert event["script"]["type"] == "text/javascript"
            assert all(isinstance(line, str) for line in event["script"]["exec"])


def test_every_request_is_a_real_operation_and_every_operation_is_covered(
    collection, openapi
) -> None:
    documented = {
        (method, path) for path, operations in openapi["paths"].items() for method in operations
    }
    used = {operation(item) for _, item in requests(collection["item"])}

    assert used <= documented
    assert used == documented  # all six public operations have a request


def test_example_bodies_pass_the_real_validation(collection) -> None:
    invalid = []
    for name, item in requests(collection["item"]):
        request = item["request"]
        if "body" not in request:
            continue
        assert request["body"]["options"]["raw"]["language"] == "json", name
        assert {"key": "Content-Type", "value": "application/json"} in request["header"], name
        submission = JobSubmission.model_validate(json.loads(request["body"]["raw"]))
        try:
            prepare_request(submission.question, submission.source_material)
        except InputError:
            invalid.append(name)

    assert invalid == ["Examples / Invalid submission (422, no research)"]


def test_variables_used_are_defined_by_the_environment(collection, environment) -> None:
    defined = {value["key"] for value in environment["values"]}
    text = COLLECTION.read_text(encoding="utf-8")
    used = set(VARIABLE.findall(text))

    assert used <= defined
    assert defined == {"baseUrl", "jobId", "cancelJobId"}
    values = {value["key"]: value["value"] for value in environment["values"]}
    assert values == {"baseUrl": "http://127.0.0.1:8000", "jobId": "", "cancelJobId": ""}


def test_the_files_hold_no_secrets_or_other_hosts() -> None:
    for path in (COLLECTION, ENVIRONMENT):
        text = path.read_text(encoding="utf-8")
        assert "postgresql" not in text.lower(), path.name
        assert "password" not in text.lower(), path.name
        assert not re.search(r"(api[_-]?key|token|secret|bearer)", text, re.IGNORECASE), path.name
        # The only URLs are the local base URL and the Postman schema reference.
        urls = set(re.findall(r"https?://[^\s\"'),]+", text))
        assert urls <= {"http://127.0.0.1:8000", SCHEMA_V21}, path.name


def test_scripts_only_check_responses_and_never_send_requests(collection) -> None:
    for name, item in requests(collection["item"]):
        for event in item.get("event", []):
            script = "\n".join(event["script"]["exec"])
            assert "pm.sendRequest" not in script, name
            assert "fetch(" not in script and "require(" not in script, name
            assert "pm.test(" in script, name


def test_submissions_save_the_job_id_they_are_followed_by(collection) -> None:
    saved = {}
    for name, item in requests(collection["item"]):
        script = "\n".join(
            line for event in item.get("event", []) for line in event["script"]["exec"]
        )
        match = re.search(r'pm\.environment\.set\("(\w+)"', script)
        if match:
            saved[name] = match.group(1)

    assert saved == {
        "Jobs / Submit research job": "jobId",
        "Jobs / Submit job to cancel": "cancelJobId",
        "Examples / Submit web-mode job (calls Parallel Search MCP)": "jobId",
    }


def test_the_readme_lists_exactly_the_real_endpoints(openapi) -> None:
    section = README.read_text(encoding="utf-8").split("## HTTP API", 1)[1].split("\n## ", 1)[0]
    listed = {
        (method.lower(), path)
        for method, path in re.findall(r"`(GET|POST) (/api/v1/[^`]*)`", section)
    }
    documented = {
        (method, path) for path, operations in openapi["paths"].items() for method in operations
    }

    assert listed == documented
    assert "http://127.0.0.1:8000/docs" in section
    assert "uv run alembic upgrade head" in section
    assert "uv run python -m researchpilot.api" in section
