"""
Comprehensive Performance and Load Testing Suite

This module provides various test scenarios for performance testing:
- Load testing: Normal and peak load scenarios
- Stress testing: Beyond peak load to find breaking points
- Soak testing: Long-running tests to find memory leaks
- Spike testing: Sudden increases in load

Usage:
    # Run normal load test (100 users, spawn rate 10)
    locust -f test_load.py --headless -u 100 -r 10 -t 5m --host=http://localhost:8000
    
    # Run stress test (1000 users, spawn rate 100)
    locust -f test_load.py --headless -u 1000 -r 100 -t 10m --host=http://your-api-url
    
    # Run specific test scenario
    locust -f test_load.py -u 100 -r 10 -t 5m --host=http://localhost:8000 LoadTestUser
"""
import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from locust import HttpUser, task, between, events, LoadTestShape, tag
from locust.runners import MasterRunner, WorkerRunner
from locust.exception import StopUser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_FILES_DIR = Path(__file__).parent.parent / "test_data"
TEST_IMAGE = TEST_FILES_DIR / "test_image.jpg"
TEST_VIDEO = TEST_FILES_DIR / "test_video.mp4"
TEST_AUDIO = TEST_FILES_DIR / "test_audio.wav"
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Test users with different roles
TEST_USERS = [
    {"username": f"user_{i}@example.com", "password": "testpass123"}
    for i in range(1, 6)  # 5 test users
]

class TestDataGenerator:
    """Helper class to generate test data"""
    
    @staticmethod
    def create_test_image():
        """Create a test image file"""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (800, 600), color='red')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Test Image {time.time()}", fill='white')
        img.save(TEST_IMAGE)
        return TEST_IMAGE
    
    @staticmethod
    def create_test_video():
        """Create a test video file"""
        import numpy as np
        import cv2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(TEST_VIDEO), fourcc, 30.0, (640, 480))
        
        for _ in range(30):  # 1 second of video at 30fps
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return TEST_VIDEO

class BaseUser(HttpUser):
    """Base user class with common functionality"""
    abstract = True
    wait_time = between(0.5, 2)  # Random wait between tasks
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_token = None
        self.user_data = random.choice(TEST_USERS)
    
    def on_start(self):
        """Login when a user starts"""
        try:
            response = self.client.post(
                "/api/v1/auth/login",
                json=self.user_data
            )
            if response.status_code == 200:
                self.auth_token = response.json().get("access_token")
            else:
                logger.error(f"Failed to login: {response.text}")
                raise StopUser()  # Stop this user if login fails
        except Exception as e:
            logger.error(f"Error during login: {e}")
            raise StopUser()
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get headers with auth token"""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}

@tag('load')
class LoadTestUser(BaseUser):
    """Simulate normal user load"""
    
    @task(5)
    def analyze_image(self):
        """Test image analysis endpoint"""
        with open(TestDataGenerator.create_test_image(), 'rb') as f:
            self.client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                headers=self.headers,
                name="/api/v1/analyze/image"
            )
    
    @task(3)
    def analyze_video(self):
        """Test video analysis endpoint"""
        with open(TestDataGenerator.create_test_video(), 'rb') as f:
            self.client.post(
                "/api/v1/analyze/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                headers=self.headers,
                name="/api/v1/analyze/video"
            )
    
    @task(2)
    def view_dashboard(self):
        """View dashboard"""
        self.client.get(
            "/dashboard",
            headers=self.headers,
            name="/dashboard"
        )

@tag('stress')
class StressTestUser(BaseUser):
    """Simulate high load conditions"""
    wait_time = between(0.1, 0.5)  # More aggressive than normal users
    
    @task(3)
    def analyze_image_heavy(self):
        """Test with larger images"""
        with open(TestDataGenerator.create_test_image(), 'rb') as f:
            self.client.post(
                "/api/v1/analyze/image",
                files={"file": ("large.jpg", f, "image/jpeg")},
                headers=self.headers,
                name="/api/v1/analyze/image[large]"
            )
    
    @task(2)
    def analyze_multiple(self):
        """Test multiple concurrent analyses"""
        with open(TestDataGenerator.create_test_image(), 'rb') as img_file, \
             open(TestDataGenerator.create_test_video(), 'rb') as vid_file:
            
            self.client.post(
                "/api/v1/analyze/batch",
                files=[
                    ("files", ("img1.jpg", img_file, "image/jpeg")),
                    ("files", ("vid1.mp4", vid_file, "video/mp4"))
                ],
                headers=self.headers,
                name="/api/v1/analyze/batch"
            )

class StagesShape(LoadTestShape):
    ""
    A custom load test shape that runs different stages of tests:
    1. Ramp up to 100 users
    2. Stay at 100 users for 5 minutes
    3. Ramp up to 500 users
    4. Stay at 500 users for 10 minutes
    5. Ramp down to 0 users
    """
    
    stages = [
        {"duration": 60, "users": 100, "spawn_rate": 10},  # Ramp up to 100 users
        {"duration": 300, "users": 100, "spawn_rate": 10},  # Stay at 100 users
        {"duration": 300, "users": 500, "spawn_rate": 50},  # Ramp up to 500 users
        {"duration": 600, "users": 500, "spawn_rate": 50},  # Stay at 500 users
        {"duration": 300, "users": 0, "spawn_rate": 10},    # Ramp down
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
            run_time -= stage["duration"]
        
        return None  # Test is complete

# Event hooks
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test data when tests start"""
    if not isinstance(environment.runner, WorkerRunner):
        # Only run on master or in standalone mode
        logger.info("Initializing test data...")
        TestDataGenerator.create_test_image()
        TestDataGenerator.create_test_video()

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response,
              context, exception, start_time, url, **kwargs):
    """Log failed requests"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    elif response and response.status_code >= 400:
        logger.error(
            f"Request failed with {response.status_code}: "
            f"{request_type} {url} - {response.text}"
        )

if __name__ == "__main__":
    import sys
    import os
    
    # Ensure test data exists
    if not os.path.exists(TEST_IMAGE):
        TestDataGenerator.create_test_image()
    if not os.path.exists(TEST_VIDEO):
        TestDataGenerator.create_test_video()
    
    print("Test data initialized. Run with: locust -f test_load.py")
    os.system("locust -f test_load.py")
