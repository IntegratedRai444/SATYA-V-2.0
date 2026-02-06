#!/usr/bin/env python3
"""
Verify ML Models are enabled everywhere
"""

import os
from pathlib import Path

def check_env_file(file_path: str, var_name: str) -> bool:
    """Check if environment variable is set to true in file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.strip().startswith(f'{var_name}='):
                    value = line.split('=', 1)[1].strip().lower()
                    return value == 'true'
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return False

def main():
    print("🔍 Verifying ML Models are Enabled Everywhere...")
    print("=" * 60)
    
    ml_vars = [
        'ENABLE_ML_MODELS',
        'ENABLE_ADVANCED_MODELS',
        'ENABLE_ML_OPTIMIZATION',
        'STRICT_MODE_ENABLED'
    ]
    
    env_files = [
        '.env',
        '.env.example', 
        '.env.production',
        'server/.env',
        'server/python/.env'
    ]
    
    all_good = True
    
    for env_file in env_files:
        if Path(env_file).exists():
            print(f"\n📁 Checking {env_file}:")
            for var in ml_vars:
                status = "✅" if check_env_file(env_file, var) else "❌"
                print(f"   {status} {var}")
                if check_env_file(env_file, var) == False:
                    all_good = False
        else:
            print(f"\n❌ {env_file} not found")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 All ML Models are ENABLED everywhere!")
        print("🚀 Ready to start with real AI deepfake detection!")
    else:
        print("⚠️ Some ML models are not enabled. Check the output above.")
    
    print("\n🔄 To apply changes, restart all services:")
    print("   npm run start:all")
    print("=" * 60)

if __name__ == "__main__":
    main()
