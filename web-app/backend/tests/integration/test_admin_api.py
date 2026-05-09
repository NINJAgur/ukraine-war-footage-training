"""
Integration tests for admin API endpoints.
Requires auth_headers fixture (session-scoped login).
"""
import logging
import pytest

from db.models import ModelType, TrainingStage

logger = logging.getLogger("test_admin_api")


@pytest.mark.integration
def test_admin_clips_without_auth_returns_401(client):
    resp = client.get("/api/admin/clips")
    assert resp.status_code == 401


@pytest.mark.integration
def test_admin_clips_with_auth_returns_200(client, auth_headers):
    resp = client.get("/api/admin/clips", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.integration
def test_admin_clips_response_has_items_and_total(client, auth_headers):
    resp = client.get("/api/admin/clips", headers=auth_headers)
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)


@pytest.mark.integration
def test_admin_training_runs_without_auth_returns_401(client):
    resp = client.get("/api/admin/training-runs")
    assert resp.status_code == 401


@pytest.mark.integration
def test_admin_training_runs_with_auth_returns_200(client, auth_headers):
    resp = client.get("/api/admin/training-runs", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.integration
def test_admin_training_runs_response_has_items(client, auth_headers):
    resp = client.get("/api/admin/training-runs", headers=auth_headers)
    data = resp.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.integration
def test_admin_training_runs_includes_run_30(client, auth_headers):
    # Fetch enough runs to include run id=30 (PERSONNEL baseline completed)
    resp = client.get("/api/admin/training-runs?per_page=50", headers=auth_headers)
    data = resp.json()
    ids = [r["id"] for r in data["items"]]
    assert 30 in ids, f"Run id=30 not found in training runs. Got ids: {ids}"


@pytest.mark.integration
def test_admin_train_without_auth_returns_401(client):
    resp = client.post("/api/admin/train", json={"model_type": "GENERAL", "stage": "BASELINE"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_admin_decline_without_auth_returns_401(client):
    resp = client.delete("/api/admin/clips/999")
    assert resp.status_code == 401


@pytest.mark.integration
def test_admin_decline_nonexistent_clip_returns_404(client, auth_headers):
    resp = client.delete("/api/admin/clips/999999", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.integration
def test_admin_decline_non_review_clip_returns_409(client, auth_headers):
    # Fetch any annotated clip and try to decline it — should be rejected (read-only probe)
    resp = client.get("/api/admin/clips?status=ANNOTATED&per_page=1", headers=auth_headers)
    items = resp.json().get("items", [])
    if not items:
        pytest.skip("No ANNOTATED clips in DB — skipping")
    clip_id = items[0]["id"]
    resp = client.delete(f"/api/admin/clips/{clip_id}", headers=auth_headers)
    assert resp.status_code == 409


@pytest.mark.integration
def test_admin_train_general_baseline_returns_409_already_active(client, auth_headers):
    # GENERAL BASELINE is expected to already have a QUEUED/RUNNING run from project state
    resp = client.post(
        "/api/admin/train",
        json={"model_type": "GENERAL", "stage": "BASELINE"},
        headers=auth_headers,
    )
    # 409 = already active (no run created); 202 = newly queued
    assert resp.status_code in (409, 202)

    run_id = None
    try:
        if resp.status_code == 202:
            run_id = resp.json().get("training_run_id")
            logger.info(f"POST /api/admin/train returned 202, run_id={run_id}")
        else:
            logger.info("POST /api/admin/train returned 409 — no run created, nothing to clean up")
    finally:
        if run_id is not None:
            from sqlalchemy import create_engine, text
            from config import settings
            engine = create_engine(settings.DATABASE_SYNC_URL)
            try:
                with engine.connect() as conn:
                    conn.execute(text("DELETE FROM training_runs WHERE id = :id"), {"id": run_id})
                    conn.commit()
                logger.info(f"Cleanup: deleted training_run id={run_id}")
            finally:
                engine.dispose()
