"""
End-to-end tests for security scenarios including rate limiting and authentication
"""
import pytest
import time
from fastapi.testclient import TestClient
from server.main import app

# Test client
client = TestClient(app)

class TestSecurityScenarios:
    """Test security-related scenarios"""

    def test_rate_limiting(self):
        """Test that rate limiting is working"""
        # Make multiple rapid requests to a rate-limited endpoint
        endpoint = "/api/analyze/status/some_id"
        responses = []
        
        # First few requests should succeed
        for _ in range(5):
            response = client.get(endpoint)
            responses.append(response.status_code)
            time.sleep(0.1)  # Small delay between requests
        
        # After rate limit is hit, should get 429 Too Many Requests
        response = client.get(endpoint)
        responses.append(response.status_code)
        
        # Verify we got rate limited
        assert 429 in responses
        assert responses.count(200) <= 5  # Should not exceed rate limit

    def test_authentication_required(self):
        """Test that protected endpoints require authentication"""
        # Try to access protected endpoint without token
        response = client.get("/api/user/profile")
        assert response.status_code == 401  # Unauthorized
        
        # Try with invalid token
        response = client.get(
            "/api/user/profile",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 403  # Forbidden

    @pytest.mark.parametrize("endpoint,method,data", [
        ("/api/analyze/image", "POST", {"file": ("test.jpg", b"test", "image/jpeg")}),
        ("/api/analyze/video", "POST", {"file": ("test.mp4", b"test", "video/mp4")}),
        ("/api/analyze/audio", "POST", {"file": ("test.wav", b"test", "audio/wav")}),
    ])
    def test_file_upload_validation(self, endpoint, method, data):
        """Test file upload validation for different file types"""
        # Test empty file
        response = client.post(
            endpoint,
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400
        assert "empty" in response.json().get("detail", "").lower()
        
        # Test invalid content type
        response = client.post(
            endpoint,
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        
        # Test valid file
        response = client.post(endpoint, files=data)
        # Should be either accepted (202) or bad request if test file is invalid
        assert response.status_code in (202, 400)

    def test_cross_origin_requests(self):
        """Test CORS headers are properly set"""
        # Make an OPTIONS request to check CORS headers
        response = client.options(
            "/api/analyze/image",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Check CORS headers
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_error_handling(self):
        """Test proper error handling for various scenarios"""
        # Test invalid endpoint
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
        
        # Test invalid HTTP method
        response = client.post("/api/analyze/image/status/123")
        assert response.status_code == 405  # Method Not Allowed
        
        # Test malformed JSON
        response = client.post(
            "/api/some_endpoint",
            content="{"invalid": json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
