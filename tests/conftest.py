"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_data.app import create_app
from eugene_plexus_data.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Pre-seed config so dataRoot / inboxDir / archiveDir land inside
    # tmp_path — otherwise the defaults would be created relative to CWD
    # and leak between test runs (and /v1/config/test would write there).
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataRoot": str(tmp_path / "data-root"),
                "inboxDir": str(tmp_path / "inbox"),
                "archiveDir": str(tmp_path / "archive"),
            }
        )
    )
    return Settings(config_file=config_path)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
