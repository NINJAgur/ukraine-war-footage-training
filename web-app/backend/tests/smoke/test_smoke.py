"""
Smoke tests: hit the LIVE server at http://localhost:8000.
Skipped gracefully if server is not running.

Run with:
    pytest -m smoke tests/smoke/test_smoke.py
"""
import pytest

_BASE = "http://localhost:8000"


def _get(path: str):
    import requests
    return requests.get(f"{_BASE}{path}", timeout=5)


def _server_reachable() -> bool:
    import requests
    try:
        requests.get(f"{_BASE}/api/stats", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.smoke
def test_stats_returns_200_within_2s():
    import time
    import requests

    try:
        start = time.monotonic()
        resp = requests.get(f"{_BASE}/api/stats", timeout=5)
        elapsed = time.monotonic() - start
    except (requests.ConnectionError, ConnectionRefusedError, OSError) as exc:
        pytest.skip(f"Server not reachable: {exc}")

    assert resp.status_code == 200
    assert elapsed < 2.0, f"Response took {elapsed:.2f}s — expected < 2s"


@pytest.mark.smoke
def test_annotated_clips_returns_200():
    import requests

    try:
        resp = requests.get(f"{_BASE}/api/annotated-clips", timeout=5)
    except (requests.ConnectionError, ConnectionRefusedError, OSError) as exc:
        pytest.skip(f"Server not reachable: {exc}")

    assert resp.status_code == 200


@pytest.mark.smoke
def test_archive_returns_200():
    import requests

    try:
        resp = requests.get(f"{_BASE}/api/archive", timeout=5)
    except (requests.ConnectionError, ConnectionRefusedError, OSError) as exc:
        pytest.skip(f"Server not reachable: {exc}")

    assert resp.status_code == 200
