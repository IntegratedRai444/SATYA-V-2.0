#!/usr/bin/env python3
"""
Complete SatyaAI Workflow Test
Tests the entire Upload → Detection → Output pipeline
"""

import requests
import json
import time
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a realistic test image for analysis."""
    # Create a face-like image with more realistic features
    face_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Skin tone background
    face_image[:, :] = [220, 180, 140]
    
    # Add facial features
    # Eyes
    face_image[80:100, 70:90] = [50, 30, 20]  # Left eye
    face_image[80:100, 134:154] = [50, 30, 20]  # Right eye
    
    # Nose
    face_image[110:140, 100:124] = [200, 160, 120]
    
    # Mouth
    face_image[150:165, 90:134] = [100, 50, 50]
    
    # Add realistic noise
    noise = np.random.randint(-15, 15, (224, 224, 3))
    face_image = np.clip(face_image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL and then to bytes
    pil_image = Image.fromarray(face_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    
    return buffer.getvalue()

def test_python_server():
    """Test Python server directly."""
    print("🐍 TESTING PYTHON SERVER")
    print("-" * 40)
    
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:5001/health', timeout=5)
        print(f"Health Check: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Status: {health_data.get('status')}")
            print(f"✅ Version: {health_data.get('version')}")
            print(f"✅ Models: {health_data.get('real_ai_models')}")
        
        # Test image analysis
        image_bytes = create_test_image()
        files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
        
        print(f"📸 Testing image analysis ({len(image_bytes)} bytes)...")
        analysis_response = requests.post(
            'http://localhost:5001/api/analyze/image', 
            files=files, 
            timeout=30
        )
        
        print(f"Analysis Response: {analysis_response.status_code}")
        
        if analysis_response.status_code == 200:
            result = analysis_response.json()
            print(f"✅ Authenticity: {result.get('authenticity')}")
            print(f"✅ Confidence: {result.get('confidence')}%")
            print(f"✅ Case ID: {result.get('case_id')}")
            print(f"✅ Processing Time: {result.get('technical_details', {}).get('processing_time_seconds', 0)}s")
            return True
        else:
            print(f"❌ Analysis failed: {analysis_response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Python server not responding")
        return False
    except Exception as e:
        print(f"❌ Python server test failed: {e}")
        return False

def test_nodejs_server():
    """Test Node.js server."""
    print("\n🟢 TESTING NODE.JS SERVER")
    print("-" * 40)
    
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:3000/api/health', timeout=5)
        print(f"Health Check: {health_response.status_code}")
        
        if health_response.status_code == 200:
            print("✅ Node.js server responding")
            
            # Test analysis endpoint through Node.js
            image_bytes = create_test_image()
            files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
            
            print(f"📸 Testing Node.js analysis endpoint...")
            analysis_response = requests.post(
                'http://localhost:3000/api/analysis/image', 
                files=files, 
                timeout=30
            )
            
            print(f"Analysis Response: {analysis_response.status_code}")
            
            if analysis_response.status_code == 200:
                result = analysis_response.json()
                print(f"✅ Node.js → Python → AI: WORKING")
                print(f"✅ Final Result: {result.get('authenticity')}")
                return True
            else:
                print(f"❌ Node.js analysis failed: {analysis_response.text[:200]}")
                return False
        else:
            print("❌ Node.js server not responding properly")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Node.js server not responding")
        return False
    except Exception as e:
        print(f"❌ Node.js server test failed: {e}")
        return False

def test_database():
    """Test database operations."""
    print("\n💾 TESTING DATABASE")
    print("-" * 40)
    
    try:
        import sqlite3
        
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        
        # Check database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"✅ Database connected")
        print(f"✅ Tables: {len(tables)}")
        
        # Check if we can query (basic test)
        try:
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            count = cursor.fetchone()[0]
            print(f"✅ Database queries working: {count} objects")
        except Exception as e:
            print(f"⚠️  Database query test: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run complete workflow test."""
    print("🚀 SATYAAI COMPLETE WORKFLOW TEST")
    print("=" * 60)
    print("Testing: Upload → Detection → Output Pipeline")
    print("=" * 60)
    
    results = {
        'python_server': False,
        'nodejs_server': False,
        'database': False,
        'ai_detection': False
    }
    
    # Test each component
    results['python_server'] = test_python_server()
    results['nodejs_server'] = test_nodejs_server()
    results['database'] = test_database()
    
    # AI detection is tested as part of Python server
    results['ai_detection'] = results['python_server']
    
    print("\n" + "=" * 60)
    print("📊 WORKFLOW TEST RESULTS")
    print("=" * 60)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        component_name = component.replace('_', ' ').title()
        print(f"{status_icon} {component_name}: {'WORKING' if status else 'NEEDS ATTENTION'}")
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"\n🎯 OVERALL STATUS: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎉 COMPLETE WORKFLOW: FULLY FUNCTIONAL!")
        print("🔥 Upload → Detection → Output: 100% OPERATIONAL")
    elif working_components >= total_components * 0.75:
        print("✅ WORKFLOW: MOSTLY FUNCTIONAL")
        print("💡 Minor fixes needed for full operation")
    else:
        print("⚠️  WORKFLOW: NEEDS ATTENTION")
        print("🔧 Several components need fixes")
    
    print("\n" + "=" * 60)
    
    return working_components == total_components

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)