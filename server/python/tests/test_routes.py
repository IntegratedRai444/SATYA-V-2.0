import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Example test for root endpoint
# def test_root():
#     response = client.get("/")
#     assert response.status_code == 200
#     assert "message" in response.json()

def test_health():
    response = client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["success"] is True 