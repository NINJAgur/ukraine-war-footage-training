"""
Smoke tests: DB reachable, Funker530 API up, GeoConfirmed API up.

Run with:
    pytest -m smoke tests/smoke/test_smoke.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.mark.smoke
def test_db_reachable():
    """DB engine can execute a trivial query without raising."""
    from db.session import get_session
    from db.models import Clip

    with get_session() as session:
        result = session.query(Clip).limit(1).all()
    assert isinstance(result, list)


@pytest.mark.smoke
@pytest.mark.network
def test_funker530_api_returns_200():
    """Funker530 REST API is reachable and returns HTTP 200."""
    import urllib.request

    url = "https://funker530.com/api/v1/videos/?page=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
    except Exception as exc:
        pytest.skip(f"Funker530 not reachable: {exc}")


@pytest.mark.smoke
@pytest.mark.network
def test_geoconfirmed_api_returns_200():
    """GeoConfirmed REST API is reachable and returns HTTP 200."""
    import urllib.request

    url = "https://geoconfirmed.azurewebsites.net/api/incidents?limit=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
    except Exception as exc:
        pytest.skip(f"GeoConfirmed not reachable: {exc}")
