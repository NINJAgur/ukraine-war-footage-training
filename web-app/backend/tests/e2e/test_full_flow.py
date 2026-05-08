"""
E2E test: full clip lifecycle via API.
login → list clips → submit → verify REVIEW → approve → verify PENDING

Run with:
    pytest -m e2e tests/e2e/test_full_flow.py
"""
import hashlib
import uuid

import pytest


def _delete_clip_by_hash(url_hash: str):
    """Synchronous teardown: delete test clip from DB using scraper-engine session."""
    import sys
    from pathlib import Path
    _scraper_root = Path(__file__).resolve().parents[4] / "scraper-engine"
    if str(_scraper_root) not in sys.path:
        sys.path.insert(0, str(_scraper_root))
    from db.session import get_session
    from db.models import Clip

    with get_session() as session:
        clip = session.query(Clip).filter_by(url_hash=url_hash).first()
        if clip:
            session.delete(clip)


@pytest.mark.e2e
def test_full_clip_lifecycle(client):
    # 1. Login
    login_resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. List clips (admin)
    list_resp = client.get("/api/admin/clips", headers=headers)
    assert list_resp.status_code == 200

    # 3. Submit new unique clip
    unique_url = f"https://example.com/e2e-test/{uuid.uuid4()}"
    url_hash = hashlib.sha256(unique_url.encode()).hexdigest()

    submit_resp = client.post("/api/submit", json={
        "url": unique_url,
        "title": "E2E test clip",
        "description": "Automated e2e test submission",
    })
    assert submit_resp.status_code == 201
    clip_data = submit_resp.json()
    clip_id = clip_data["id"]

    try:
        # 4. Verify it appears in archive with REVIEW status
        archive_resp = client.get("/api/archive?status=REVIEW&per_page=100")
        assert archive_resp.status_code == 200
        archive_items = archive_resp.json()["items"]
        review_ids = [item["id"] for item in archive_items]
        assert clip_id in review_ids, f"Submitted clip {clip_id} not found in REVIEW archive"

        # 5. Admin approve it
        approve_resp = client.post(f"/api/admin/clips/{clip_id}/approve", headers=headers)
        assert approve_resp.status_code == 200

        # 6. Verify status changed to PENDING
        archive_resp2 = client.get("/api/archive?status=PENDING&per_page=100")
        assert archive_resp2.status_code == 200
        pending_items = archive_resp2.json()["items"]
        pending_ids = [item["id"] for item in pending_items]
        assert clip_id in pending_ids, f"Approved clip {clip_id} not found in PENDING archive"

    finally:
        _delete_clip_by_hash(url_hash)
