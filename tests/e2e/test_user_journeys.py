"""
End-to-end tests for user journeys and negative test cases
"""
import pytest
import os
import time
import json
from pathlib import Path
from fastapi.testclient import TestClient
from server.main import app

# Test client
client = TestClient(app)

# Test file paths
TEST_FILES_DIR = Path(__file__).parent / "test_files"
TEST_IMAGE = TEST_FILES_DIR / "test_face.jpg"
TEST_VIDEO = TEST_FILES_DIR / "test_video.mp4"
TEST_AUDIO = TEST_FILES_DIR / "test_audio.wav"

class TestUserJourneys:
    """Test complete user journeys and negative scenarios"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Ensure test files directory exists
        TEST_FILES_DIR.mkdir(exist_ok=True)
        
        # Create test files if they don't exist
        for test_file in [TEST_IMAGE, TEST_VIDEO, TEST_AUDIO]:
            if not test_file.exists():
                test_file.write_bytes(b"test content")
        
        yield
        
        # Cleanup (if needed)
        pass

    def test_complete_image_analysis_journey(self):
        """Test complete user journey for image analysis"""
        # 1. Upload image
        with open(TEST_IMAGE, "rb") as f:
            response = client.post(
                "/api/analyze/image",
                files={"file": ("test_face.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 202
        job_id = response.json().get("job_id")
        assert job_id is not None

        # 2. Check status until completion
        for _ in range(10):  # Max 10 attempts
            status_response = client.get(f"/api/analyze/{job_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail("Analysis failed")
            
            time.sleep(1)  # Wait before next status check
        else:
            pytest.fail("Analysis timed out")

        # 3. Get results
        result_response = client.get(f"/api/analyze/{job_id}/results")
        assert result_response.status_code == 200
        results = result_response.json()
        
        # 4. Verify results
        assert "analysis_id" in results
        assert "status" in results
        assert results["status"] == "completed"
        assert "results" in results
        assert "authenticity_score" in results["results"]

    def test_invalid_file_upload(self):
        """Test uploading invalid file types"""
        # Test unsupported file type
        response = client.post(
            "/api/analyze/image",
            files={"file": ("invalid.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json().get("detail", "")

    def test_missing_file_upload(self):
        """Test request with missing file"""
        response = client.post("/api/analyze/image")
        assert response.status_code == 422  # Validation error

    @pytest.mark.parametrize("endpoint,file_param", [
        ("/api/analyze/image", "file"),
        ("/api/analyze/video", "video"),
        ("/api/analyze/audio", "audio")
    ])
    def test_large_file_upload(self, endpoint, file_param):
        """Test uploading files that are too large"""
        # Create a large file (5MB)
        large_file = TEST_FILES_DIR / "large_file.bin"
        large_file.write_bytes(b"0" * (5 * 1024 * 1024))  # 5MB
        
        try:
            with open(large_file, "rb") as f:
                response = client.post(
                    endpoint,
                    files={file_param: ("large_file.bin", f, "application/octet-stream")}
                )
            assert response.status_code == 413  # Payload Too Large
        finally:
            if large_file.exists():
                large_file.unlink()

    def test_concurrent_uploads(self):
        """Test handling of concurrent file uploads"""
        import threading
        
        results = {}
        
        def upload_file(index):
            with open(TEST_IMAGE, "rb") as f:
                response = client.post(
                    "/api/analyze/image",
                    files={"file": (f"test_{index}.jpg", f, "image/jpeg")}
                )
            results[index] = response
        
        # Start multiple upload threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=upload_file, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all uploads were successful
        for i in range(3):
            assert results[i].status_code == 202
            assert "job_id" in results[i].json()

    def test_analysis_status_not_found(self):
        """Test checking status of non-existent analysis"""
        response = client.get("/api/analyze/nonexistent_id/status")
        assert response.status_code == 404

    def test_results_before_completion(self):
        """Test getting results before analysis is complete"""
        # Start an analysis
        with open(TEST_IMAGE, "rb") as f:
            response = client.post(
                "/api/analyze/image",
                files={"file": ("test_face.jpg", f, "image/jpeg")}
            )
        job_id = response.json().get("job_id")
        
        # Immediately try to get results
        response = client.get(f"/api/analyze/{job_id}/results")
        assert response.status_code in [202, 400]  # Should not be 200
        
        # Clean up - wait for the analysis to complete
        for _ in range(10):
            status_response = client.get(f"/api/analyze/{job_id}/status")
            if status_response.json().get("status") == "completed":
                break
            time.sleep(1)
