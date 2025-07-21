#!/usr/bin/env python3
"""
Simple test script to verify Flask server endpoints
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Example test for health endpoint
# def test_health():
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json()["success"] is True 