"""
Integration tests for /api/auth/login endpoint.
"""
import pytest


@pytest.mark.integration
def test_login_valid_credentials_returns_200(client):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200


@pytest.mark.integration
def test_login_returns_access_token(client):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    data = resp.json()
    assert "access_token" in data
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0


@pytest.mark.integration
def test_login_wrong_password_returns_401(client):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_login_wrong_username_returns_401(client):
    resp = client.post("/api/auth/login", json={"username": "notadmin", "password": "admin123"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_login_empty_credentials_returns_401(client):
    resp = client.post("/api/auth/login", json={"username": "", "password": ""})
    assert resp.status_code == 401


@pytest.mark.integration
def test_login_token_is_nonempty_string(client):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    token = resp.json().get("access_token", "")
    assert len(token) > 10  # JWT tokens are always much longer than 10 chars
