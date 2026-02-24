#!/usr/bin/env python3
"""
Full Workflow Test for SatyaAI
Tests the complete deepfake detection pipeline with real face dataset
"""

import os
import requests
import json
import time
import random
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:5001/api/v2"
DATASET_PATH = r"E:\SATYA-V-2.0\real_and_fake_face"
CREDENTIALS = {
    "email": "Rishabhkapoor@atomicmail.io",
    "password": "Rishabhkapoor@0444"
}

class WorkflowTester:
    def __init__(self):
        self.session = requests.Session()
        # Enable cookie handling explicitly
        self.session.cookies.clear()
        self.auth_token = None
        self.test_results = []
        
    def login(self) -> bool:
        """Authenticate with the API"""
        print("🔐 Attempting login...")
        
        try:
            response = self.session.post(
                f"{API_BASE_URL}/auth/login",
                json=CREDENTIALS,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"🔍 Login response: {data}")
                if data.get("success"):
                    # Check if we received cookies (HTTP-only authentication)
                    cookies = self.session.cookies.get_dict()
                    print(f"🔍 Cookies received: {list(cookies.keys())}")
                    
                    # For cookie-based auth, we don't need to set Authorization header
                    # The session will automatically send cookies
                    print("✅ Login successful (cookie-based)")
                    return True
                else:
                    print(f"❌ Login failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Login failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return False
    
    def upload_and_analyze(self, image_path: Path, expected_label: str) -> Dict[str, Any]:
        """Upload image for analysis and return results"""
        print(f"📤 Testing {expected_label} image: {image_path.name}")
        
        try:
            # Upload file
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                # For cookie-based auth, we don't need Authorization header
                # The session automatically sends cookies
                print(f"🔍 Using cookie-based authentication")
                print(f"🔍 Session cookies: {dict(self.session.cookies)}")
                response = self.session.post(
                    f"{API_BASE_URL}/analysis/image",
                    files=files
                )
            
            print(f"🔍 Upload response status: {response.status_code}")
            if response.status_code != 202:
                try:
                    error_data = response.json()
                    print(f"🔍 Upload error response: {error_data}")
                except:
                    print(f"🔍 Upload error text: {response.text}")
                error_msg = f"Upload failed: {response.status_code}"
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            upload_data = response.json()
            if not upload_data.get("success"):
                error_msg = upload_data.get("error", "Upload failed")
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            job_id = upload_data.get("job_id")
            print(f"🆔 Job created: {job_id}")
            
            # Poll for results
            result = self.poll_for_results(job_id)
            result["expected_label"] = expected_label
            result["actual_label"] = result.get("classification", "unknown")
            result["correct"] = result["actual_label"] == expected_label
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def poll_for_results(self, job_id: str, max_wait: int = 120) -> Dict[str, Any]:
        """Poll for analysis results"""
        print(f"⏳ Polling for job {job_id}...")
        
        start_time = time.time()
        poll_interval = 2
        
        while time.time() - start_time < max_wait:
            try:
                # For cookie-based auth, we don't need Authorization header
                response = self.session.get(
                    f"{API_BASE_URL}/results/{job_id}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "completed":
                        print("✅ Analysis completed")
                        return data
                    elif status == "failed":
                        error = data.get("error", "Analysis failed")
                        print(f"❌ {error}")
                        return {"status": "failed", "error": error}
                    elif status == "processing":
                        print(f"⏳ Still processing... ({int(time.time() - start_time)}s)")
                    else:
                        print(f"⚠️ Unknown status: {status}")
                
                elif response.status_code == 404:
                    print(f"❌ Job {job_id} not found (404)")
                    return {"status": "not_found", "error": "Job not found"}
                else:
                    print(f"⚠️ Poll error: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Poll error: {str(e)}")
            
            time.sleep(poll_interval)
        
        return {"status": "timeout", "error": "Analysis timeout"}
    
    def test_dataset(self, num_samples: int = 20):
        """Test a sample of images from the dataset"""
        print(f"🧪 Testing {num_samples} random images from dataset...")
        
        # Get real and fake images
        real_path = Path(DATASET_PATH) / "training_real"
        fake_path = Path(DATASET_PATH) / "training_fake"
        
        # Get all files in directories
        real_images = []
        fake_images = []
        
        if real_path.exists():
            real_images = list(real_path.glob("*.jpg")) + list(real_path.glob("*.jpeg")) + list(real_path.glob("*.png"))
        
        if fake_path.exists():
            fake_images = list(fake_path.glob("*.jpg")) + list(fake_path.glob("*.jpeg")) + list(fake_path.glob("*.png"))
        
        # Sample images
        real_images = real_images[:num_samples//2]
        fake_images = fake_images[:num_samples//2]
        
        test_images = real_images + fake_images
        random.shuffle(test_images)
        
        results = {
            "total_tests": len(test_images),
            "real_correct": 0,
            "fake_correct": 0,
            "real_total": len(real_images),
            "fake_total": len(fake_images),
            "results": []
        }
        
        for i, image_path in enumerate(test_images):
            expected_label = "real" if "real_" in image_path.name else "fake"
            result = self.upload_and_analyze(image_path, expected_label)
            
            if result["success"]:
                if expected_label == "real" and result["correct"]:
                    results["real_correct"] += 1
                elif expected_label == "fake" and result["correct"]:
                    results["fake_correct"] += 1
                    
                status = "✅" if result["correct"] else "❌"
                print(f"{i+1}: {status} {expected_label.upper()} - {result.get('confidence', 0):.2f}")
            else:
                print(f"{i+1}: ❌ {expected_label.upper()} - ERROR: {result.get('error', 'Unknown')}")
            
            results["results"].append(result)
            
            # Small delay between tests
            time.sleep(1)
        
        # Calculate accuracy
        total_correct = results["real_correct"] + results["fake_correct"]
        accuracy = (total_correct / results["total_tests"]) * 100 if results["total_tests"] > 0 else 0
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Real Images: {results['real_total']} (Correct: {results['real_correct']})")
        print(f"Fake Images: {results['fake_total']} (Correct: {results['fake_correct']})")
        print(f"Overall Accuracy: {accuracy:.1f}%")
        
        return results
    
    def run_full_test(self):
        """Run the complete workflow test"""
        print("🚀 Starting Full Workflow Test")
        print("=" * 50)
        
        # Test login
        if not self.login():
            print("❌ Cannot proceed without authentication")
            return
        
        # Test dataset
        results = self.test_dataset(10)  # Test 10 images (5 real, 5 fake)
        
        print("\n" + "=" * 50)
        print("🏁 Full Workflow Test Complete")
        
        return results

def main():
    """Main test runner"""
    print("SatyaAI Full Workflow Test")
    print("Testing deepfake detection with real face dataset")
    print()
    
    tester = WorkflowTester()
    results = tester.run_full_test()
    
    # Save results
    try:
        with open("workflow_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to workflow_test_results.json")
    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")

if __name__ == "__main__":
    main()
