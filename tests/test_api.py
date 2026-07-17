from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_traceability_invalid_document():
    response = client.get("/traceability-matrix/999999")
    assert response.status_code == 200


def test_requirement_not_found():
    response = client.get("/requirements/REQ-99999")
    assert response.status_code == 404