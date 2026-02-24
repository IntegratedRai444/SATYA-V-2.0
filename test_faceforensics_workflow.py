#!/usr/bin/env python3
"""
FaceForensics Dataset Test for SatyaAI
Tests the deepfake detection pipeline with the FaceForensics++ dataset
"""

import os
import requests
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List

# Configuration
API_BASE_URL = "http://localhost:5001/api/v2"
DATASET_PATH = r"E:\SATYA-V-2.0\FaceForensics\dataset"
CREDENTIALS = {
    "email": "Rishabhkapoor@atomicmail.io",
    "password": "Rishabhkapoor@0444"
}

class FaceForensicsTester:
    def __init__(self):
        self.session = requests.Session()
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
                if data.get("success"):
                    self.auth_token = data.get("token")
                    self.session.headers.update({
                        "Authorization": f"Bearer {self.auth_token}"
                    })
                    print("✅ Login successful")
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
    
    def get_test_images(self, dataset_type: str, num_samples: int = 10) -> List[Path]:
        """Get test images from specific dataset"""
        dataset_path = Path(DATASET_PATH) / dataset_type
        
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return []
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(dataset_path.glob(ext))
        
        if not image_files:
            print(f"❌ No images found in {dataset_type}")
            return []
        
        # Randomly sample images
        random.shuffle(image_files)
        return image_files[:num_samples]
    
    def upload_and_analyze(self, image_path: Path, expected_label: str) -> Dict[str, Any]:
        """Upload image for analysis and return results"""
        print(f"📤 Testing {expected_label} image: {image_path.name}")
        
        try:
            # Upload file
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = self.session.post(
                    f"{API_BASE_URL}/analysis/image",
                    files=files,
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
            
            if response.status_code != 202:
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
                response = self.session.get(
                    f"{API_BASE_URL}/results/{job_id}",
                    headers={"Authorization": f"Bearer {self.auth_token}"}
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
    
    def test_dataset_types(self, num_samples: int = 10):
        """Test different dataset types"""
        print("🧪 Testing FaceForensics Dataset Types")
        print("=" * 50)
        
        # Test configurations
        test_configs = [
            {
                "name": "DeepFakeDetection",
                "description": "Original deepfake videos",
                "expected_label": "fake"
            },
            {
                "name": "DeepFakes", 
                "description": "Face2Face deepfakes",
                "expected_label": "fake"
            },
            {
                "name": "FaceShifter",
                "description": "FaceShifter manipulations", 
                "expected_label": "fake"
            },
            {
                "name": "FaceSwapKowalski",
                "description": "FaceSwap manipulations",
                "expected_label": "fake"
            },
            {
                "name": "NeuralTextures",
                "description": "NeuralTextures manipulations",
                "expected_label": "fake"
            }
        ]
        
        all_results = {}
        
        for config in test_configs:
            print(f"\n🎯 Testing {config['name']} Dataset")
            print(f"📝 {config['description']}")
            print("-" * 30)
            
            images = self.get_test_images(config["name"], num_samples)
            if not images:
                continue
            
            results = []
            correct_count = 0
            
            for i, image_path in enumerate(images):
                result = self.upload_and_analyze(image_path, config["expected_label"])
                
                if result["success"]:
                    if result["correct"]:
                        correct_count += 1
                    status = "✅" if result["correct"] else "❌"
                    confidence = result.get("confidence", 0)
                    print(f"{i+1}: {status} {config['expected_label'].upper()} - Confidence: {confidence:.2f}")
                else:
                    print(f"{i+1}: ❌ ERROR: {result.get('error', 'Unknown error')}")
                
                results.append(result)
                time.sleep(1)  # Small delay between tests
            
            # Calculate accuracy for this dataset type
            total_tests = len(results)
            total_correct = sum(1 for r in results if r.get("correct", False))
            accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
            
            all_results[config["name"]] = {
                "total_tests": total_tests,
                "correct": total_correct,
                "accuracy": accuracy,
                "results": results
            }
            
            print(f"\n📊 {config['name']} Results:")
            print(f"Total Tests: {total_tests}")
            print(f"Correct: {total_correct}")
            print(f"Accuracy: {accuracy:.1f}%")
            print("-" * 30)
        
        return all_results
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all dataset types"""
        print("🚀 Starting FaceForensics Comprehensive Test")
        print("=" * 60)
        
        # Test login
        if not self.login():
            print("❌ Cannot proceed without authentication")
            return
        
        # Test all dataset types
        results = self.test_dataset_types(5)  # 5 samples from each type
        
        # Overall summary
        print("\n" + "=" * 60)
        print("🏁 OVERALL RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = sum(config["total_tests"] for config in results.values())
        total_correct = sum(config["correct"] for config in results.values())
        overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Total Correct: {total_correct}")
        print(f"🎯 Overall Accuracy: {overall_accuracy:.1f}%")
        
        # Dataset breakdown
        for name, config in results.items():
            print(f"\n📊 {name}:")
            print(f"  Tests: {config['total_tests']}")
            print(f"  Correct: {config['correct']}")
            print(f"  Accuracy: {config['accuracy']:.1f}%")
        
        # Save results
        try:
            with open("faceforensics_test_results.json", "w") as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "overall": {
                        "total_tests": total_tests,
                        "correct": total_correct,
                        "accuracy": overall_accuracy
                    },
                    "datasets": results
                }, f, indent=2)
            print(f"\n📄 Results saved to faceforensics_test_results.json")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        return results

def main():
    """Main test runner"""
    print("SatyaAI FaceForensics Dataset Test")
    print("Testing deepfake detection with comprehensive FaceForensics++ dataset")
    print()
    
    tester = FaceForensicsTester()
    results = tester.run_comprehensive_test()
    
    print("\n🏁 FaceForensics Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
