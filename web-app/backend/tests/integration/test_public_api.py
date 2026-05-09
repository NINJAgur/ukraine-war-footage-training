"""
Integration tests for public API endpoints.
Uses TestClient — requires live DB connection.
"""
import logging
import uuid

import pytest

logger = logging.getLogger("test_public_api")


def _delete_clip_by_id(clip_id: int) -> None:
    """Delete a test clip directly from DB by id."""
    from sqlalchemy import create_engine, text
    from config import settings
    engine = create_engine(settings.DATABASE_SYNC_URL)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM clips WHERE id = :id"), {"id": clip_id})
        conn.commit()
    engine.dispose()
    logger.info(f"Cleanup: deleted clip id={clip_id}")


@pytest.mark.integration
def test_stats_returns_200(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200


@pytest.mark.integration
def test_stats_has_models_dict_with_4_keys(client):
    resp = client.get("/api/stats")
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 4
    for key in ("AIRCRAFT", "VEHICLE", "PERSONNEL", "GENERAL"):
        assert key in data["models"]


@pytest.mark.integration
def test_stats_has_clips_total(client):
    resp = client.get("/api/stats")
    data = resp.json()
    assert "clips_total" in data
    assert isinstance(data["clips_total"], int)


@pytest.mark.integration
def test_annotated_clips_returns_200(client):
    resp = client.get("/api/annotated-clips")
    assert resp.status_code == 200


@pytest.mark.integration
def test_annotated_clips_is_list(client):
    resp = client.get("/api/annotated-clips")
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.integration
def test_annotated_clips_items_have_required_fields(client):
    resp = client.get("/api/annotated-clips")
    items = resp.json()
    for item in items:
        assert "videoUrl" in item
        assert "detClass" in item


@pytest.mark.integration
def test_archive_returns_200(client):
    resp = client.get("/api/archive?page=1&per_page=5")
    assert resp.status_code == 200


@pytest.mark.integration
def test_archive_has_items_total_page(client):
    resp = client.get("/api/archive?page=1&per_page=5")
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert data["page"] == 1


@pytest.mark.integration
def test_feed_returns_200(client):
    resp = client.get("/api/feed")
    assert resp.status_code == 200


@pytest.mark.integration
def test_submit_new_url_returns_201_or_409(client):
    url = f"https://example.com/test/{uuid.uuid4()}"
    clip_id = None
    try:
        resp = client.post("/api/submit", json={"url": url, "title": "Test clip"})
        assert resp.status_code in (201, 409)
        if resp.status_code == 201:
            clip_id = resp.json()["id"]
            logger.info(f"Submitted clip id={clip_id} url={url}")
    finally:
        if clip_id is not None:
            _delete_clip_by_id(clip_id)


@pytest.mark.integration
def test_submit_duplicate_returns_409(client):
    url = f"https://example.com/duplicate-test/{uuid.uuid4()}"
    clip_id = None
    try:
        r1 = client.post("/api/submit", json={"url": url, "title": "First"})
        assert r1.status_code == 201
        clip_id = r1.json()["id"]
        logger.info(f"Submitted clip id={clip_id}")
        r2 = client.post("/api/submit", json={"url": url, "title": "Second"})
        assert r2.status_code == 409
        logger.info("Duplicate correctly returned 409")
    finally:
        if clip_id is not None:
            _delete_clip_by_id(clip_id)
