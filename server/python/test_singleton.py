#!/usr/bin/env python3
"""Test singleton pattern for model preloader"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_preloader import get_detector_status

if __name__ == "__main__":
    print("Testing singleton pattern...")
    status = get_detector_status()
    print(f"Detector status: {status}")
    print("✅ Singleton test completed")
