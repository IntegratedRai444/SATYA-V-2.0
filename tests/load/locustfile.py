"""
Load testing for SatyaAI using Locust
"""
from locust import HttpUser, task, between, TaskSet, constant
import random
import os
from pathlib import Path

class UserBehavior(TaskSet):
    """Define user behavior for load testing"""
    
    def on_start(self):
        """Login when the test starts"""
        self.login()
    
    def login(self):
        """Helper method to login"""
        response = self.client.post("/api/auth/login", json={
            "username": os.getenv("TEST_USERNAME", "testuser"),
            "password": os.getenv("TEST_PASSWORD", "testpass123")
        })
        
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.interrupt()
    
    @task(3)
    def view_dashboard(self):
        """View dashboard"""
        self.client.get("/dashboard", headers=self.headers)
    
    @task(2)
    def view_recent_analyses(self):
        """View recent analyses"""
        self.client.get("/api/analyses/recent", headers=self.headers)
    
    @task(1)
    def upload_and_analyze(self):
        """Upload and analyze a file"""
        # Choose a random test file
        test_files = [
            "test_face.jpg",
            "test_video.mp4",
            "test_audio.wav"
        ]
        
        file_path = Path(__file__).parent.parent / "test_files" / random.choice(test_files)
        
        if not file_path.exists():
            return
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            
            # Start analysis
            response = self.client.post(
                "/api/analyze/multimodal",
                files=files,
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 202:
                job_id = response.json().get("job_id")
                if job_id:
                    # Check status (simulate user waiting)
                    for _ in range(3):
                        self.client.get(
                            f"/api/analyze/{job_id}/status",
                            headers=self.headers
                        )
                        self.wait()


class WebsiteUser(HttpUser):
    """Main test class"""
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # Wait between 1-5 seconds between tasks
    
    def on_start(self):
        """On start, just login"""
        pass

# Run with: locust -f tests/load/locustfile.py
