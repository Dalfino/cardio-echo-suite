"""Tests for review UI service."""
from fastapi.testclient import TestClient
from app.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "cardio-echo-suite" in r.text
    assert "Upload study" in r.text


def test_static_css():
    client = TestClient(app)
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "font-family" in r.text
