"""Shared fixtures for HTTP/API-layer tests (tests/test_api_*.py).

Business-logic tests (the rest of the suite) construct engines/repositories
directly and never touch this file. API tests need a real, running FastAPI
app — including the full composition-root lifespan (embedder, vector store,
graph repository, entity extractors, reranker) — so they exercise request
validation, dependency wiring, and response serialization the same way a
real client would, which is exactly the gap these tests close (see the
v1.0.0 audit's H2 finding).

The app is built once per test session (paying the ~20-30s fallback-detection
startup cost a single time, not once per test file) against temporary,
session-scoped storage — never the developer's real `ikios_*.db` files or
`.qdrant_local` directory — so running this suite never pollutes a real dev
environment's data, and repeated runs start from a clean slate.
"""

# Windows-specific DLL load-order bug (M9): importing `neo4j` before `torch`
# in the same process reliably breaks torch's native DLL init on this
# platform (`OSError: [WinError 1114] ... c10.dll`) — whichever library's
# test file pytest happens to collect first otherwise decides this by
# accident. conftest.py is always imported before any test module in its
# directory, so importing torch here first fixes it deterministically for
# the whole suite, regardless of collection order. See app/main.py for the
# same fix applied to the running app.
import torch  # noqa: F401

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings

_ENV_KEYS = [
    "METADATA_DB_URL",
    "LEARNING_DB_URL",
    "EXPERIMENT_DB_URL",
    "ML_REGISTRY_DB_URL",
    "ML_MODELS_DIR",
    "QDRANT_LOCAL_PATH",
    "VISION_DB_URL",
    "VISION_IMAGES_DIR",
]


@pytest.fixture(scope="session")
def client():
    tmp_dir = tempfile.mkdtemp(prefix="ikios_api_test_")
    overrides = {
        "METADATA_DB_URL": f"sqlite+aiosqlite:///{tmp_dir}/metadata.db",
        "LEARNING_DB_URL": f"sqlite+aiosqlite:///{tmp_dir}/learning.db",
        "EXPERIMENT_DB_URL": f"sqlite+aiosqlite:///{tmp_dir}/experiments.db",
        "ML_REGISTRY_DB_URL": f"sqlite+aiosqlite:///{tmp_dir}/ml_registry.db",
        "ML_MODELS_DIR": os.path.join(tmp_dir, "ml_models"),
        "QDRANT_LOCAL_PATH": os.path.join(tmp_dir, "qdrant_local"),
        "VISION_DB_URL": f"sqlite+aiosqlite:///{tmp_dir}/vision.db",
        "VISION_IMAGES_DIR": os.path.join(tmp_dir, "vision_images"),
    }
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    os.environ.update(overrides)
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
