# requires: pip install pytest httpx
import sys
from pathlib import Path

# Add backend root so `main`, `api`, `db`, `schemas`, `config` are importable
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Shared models live in repo root
_REPO_ROOT = _BACKEND_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers(client):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure function tests")
    config.addinivalue_line("markers", "integration: requires live DB")
    config.addinivalue_line("markers", "e2e: full stack flow")
    config.addinivalue_line("markers", "smoke: hits live server")
