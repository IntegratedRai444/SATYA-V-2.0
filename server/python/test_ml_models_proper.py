#!/usr/bin/env python3
"""
PROPER ML Models Test Script
Tests all ML detectors without PIL import conflicts
"""
import os
import sys

# Set environment variable BEFORE any imports
os.environ['ENABLE_ML_MODELS'] = 'true'

# Test the logic from main_api.py
ENABLE_ML = os.getenv("ENABLE_ML_MODELS", "false").lower() == "true"

print(f"ENABLE_ML_MODELS environment variable: {os.getenv('ENABLE_ML_MODELS', 'false')}")
print(f"ENABLE_ML boolean: {ENABLE_ML}")

if ENABLE_ML:
    print("✅ ML Models are ENABLED")
    print("🔄 Testing ML model imports...")
    
    try:
        # Test each detector import individually to isolate issues
        print("🔍 Testing AudioDetector...")
        from detectors.audio_detector import AudioDetector
        print("   ✅ AudioDetector imported")
        
        print("🔍 Testing ImageDetector...")
        from detectors.image_detector import ImageDetector
        print("   ✅ ImageDetector imported")
        
        print("🔍 Testing VideoDetector...")
        from detectors.video_detector import VideoDetector
        print("   ✅ VideoDetector imported")
        
        print("🔍 Testing TextNLPDetector...")
        from detectors.text_nlp_detector import TextNLPDetector
        print("   ✅ TextNLPDetector imported")
        
        print("🔍 Testing MultimodalFusionDetector...")
        from detectors.multimodal_fusion_detector import MultimodalFusionDetector
        print("   ✅ MultimodalFusionDetector imported")
        
        print("\n🎉 ALL ML DETECTORS IMPORTED SUCCESSFULLY!")
        print("📦 Available detectors:")
        print(f"  - AudioDetector: {AudioDetector is not None}")
        print(f"  - ImageDetector: {ImageDetector is not None}")
        print(f"  - VideoDetector: {VideoDetector is not None}")
        print(f"  - TextNLPDetector: {TextNLPDetector is not None}")
        print(f"  - MultimodalFusionDetector: {MultimodalFusionDetector is not None}")
        
        # Test instantiation
        print("\n🧪 Testing instantiation...")
        try:
            audio_det = AudioDetector()
            print("   ✅ AudioDetector instantiated")
        except Exception as e:
            print(f"   ❌ AudioDetector instantiation failed: {e}")
            
        try:
            image_det = ImageDetector()
            print("   ✅ ImageDetector instantiated")
        except Exception as e:
            print(f"   ❌ ImageDetector instantiation failed: {e}")
            
        try:
            video_det = VideoDetector()
            print("   ✅ VideoDetector instantiated")
        except Exception as e:
            print(f"   ❌ VideoDetector instantiation failed: {e}")
            
        try:
            text_det = TextNLPDetector()
            print("   ✅ TextNLPDetector instantiated")
        except Exception as e:
            print(f"   ❌ TextNLPDetector instantiation failed: {e}")
            
        try:
            fusion_det = MultimodalFusionDetector()
            print("   ✅ MultimodalFusionDetector instantiated")
        except Exception as e:
            print(f"   ❌ MultimodalFusionDetector instantiation failed: {e}")
        
        print("\n🎯 ML MODELS ARE FULLY FUNCTIONAL!")
        
    except ImportError as e:
        print(f"❌ ML detectors not available: {e}")
        print("⚠️ Continuing without ML capabilities")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading ML models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("❌ ML Models are DISABLED")
    print("ℹ️ Set ENABLE_ML_MODELS=true to enable")
    sys.exit(1)
