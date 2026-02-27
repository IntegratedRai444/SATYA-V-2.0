"""
SatyaAI Python Service Startup Validator
Validates all dependencies and models before starting the service
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# Load environment variables before validation
from dotenv import load_dotenv
load_dotenv()
load_dotenv('../../.env')

logger = logging.getLogger(__name__)

class StartupValidator:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success: List[str] = []

    def validate_dependencies(self) -> bool:
        """Validate all required dependencies"""
        logger.info("🔍 Validating Python dependencies...")
        
        dependencies = {
            'torch': 'PyTorch ML framework',
            'torchvision': 'PyTorch vision models',
            'cv2': 'OpenCV for image processing',
            'librosa': 'Audio processing',
            'soundfile': 'Audio file handling',
            # 'transformers': 'HuggingFace transformers', # Temporarily disabled due to circular import
            'numpy': 'Numerical computing',
            'fastapi': 'Web framework',
            'uvicorn': 'ASGI server',
            'PIL': 'Image processing'
        }
        
        all_valid = True
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                self.success.append(f"✅ {dep} - {description}")
            except ImportError as e:
                self.errors.append(f"❌ {dep} - {description} - MISSING: {e}")
                all_valid = False
        
        return all_valid

    def validate_models(self) -> bool:
        """Validate model files exist and can be loaded"""
        logger.info("🔍 Validating ML models...")
        
        model_paths = [
            'models/xception/model.pth',
            # Add other model paths as needed
        ]
        
        all_valid = True
        for model_path in model_paths:
            if Path(model_path).exists():
                try:
                    import torch
                    model = torch.load(model_path, map_location='cpu')
                    self.success.append(f"✅ Model {model_path} - {type(model)}")
                except Exception as e:
                    self.errors.append(f"❌ Model {model_path} - FAILED TO LOAD: {e}")
                    all_valid = False
            else:
                self.errors.append(f"❌ Model {model_path} - FILE NOT FOUND")
                all_valid = False
        
        return all_valid

    def validate_gpu(self) -> bool:
        """Validate GPU availability and configuration"""
        logger.info("🔍 Validating GPU configuration...")
        
        try:
            import torch
            if torch.cuda.is_available():
                self.success.append(f"✅ CUDA available - {torch.cuda.device_count()} devices")
                return True
            else:
                self.warnings.append("⚠️ CUDA not available - using CPU")
                return True  # CPU is acceptable
        except Exception as e:
            self.errors.append(f"❌ GPU validation failed: {e}")
            return False

    def validate_environment(self) -> bool:
        """Validate environment variables"""
        logger.info("🔍 Validating environment configuration...")
        
        required_env = [
            'DATABASE_URL',
            'ENABLE_ML_MODELS'
        ]
        
        all_valid = True
        for env_var in required_env:
            if os.getenv(env_var):
                self.success.append(f"✅ {env_var} - Set")
            else:
                self.warnings.append(f"⚠️ {env_var} - Not set (may cause issues)")
        
        return True  # Environment warnings are not critical

    def run_full_validation(self) -> bool:
        """Run complete startup validation"""
        logger.info("🚀 Starting SatyaAI Python Service Validation...")
        
        results = {
            'dependencies': self.validate_dependencies(),
            'models': self.validate_models(),
            'gpu': self.validate_gpu(),
            'environment': self.validate_environment()
        }
        
        # Print results
        print("\n" + "="*60)
        print("🔍 SATYAI PYTHON SERVICE VALIDATION RESULTS")
        print("="*60)
        
        if self.success:
            print("\n✅ SUCCESSFUL VALIDATIONS:")
            for item in self.success:
                print(f"   {item}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for item in self.warnings:
                print(f"   {item}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for item in self.errors:
                print(f"   {item}")
        
        print("\n" + "="*60)
        
        # Determine if startup should proceed
        critical_failures = not results['dependencies'] or not results['models']
        
        if critical_failures:
            print("🚫 CRITICAL FAILURES DETECTED - SERVICE WILL NOT START")
            print("   Fix the above errors before starting the service")
            return False
        else:
            print("✅ VALIDATION PASSED - SERVICE READY TO START")
            return True

def validate_startup() -> bool:
    """Main validation function"""
    validator = StartupValidator()
    return validator.run_full_validation()

if __name__ == "__main__":
    success = validate_startup()
    sys.exit(0 if success else 1)
