#!/usr/bin/env python3
"""
SatyaAI ML Models Enabler
Force enables ML models across all components
"""

import os
import sys
from pathlib import Path

def set_env_var(file_path: Path, var_name: str, value: str):
    """Set or update environment variable in file"""
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Find and replace or add the variable
    var_found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{var_name}='):
            lines[i] = f'{var_name}={value}'
            var_found = True
            break
    
    if not var_found:
        lines.append(f'{var_name}={value}')
    
    file_path.write_text('\n'.join(lines))
    print(f"✅ Set {var_name}={value} in {file_path}")

def main():
    print("🚀 Enabling ML Models Everywhere in SatyaAI...")
    print("=" * 60)
    
    # Environment variables to set
    ml_vars = {
        'ENABLE_ML_MODELS': 'true',
        'ENABLE_ADVANCED_MODELS': 'true', 
        'ENABLE_ML_OPTIMIZATION': 'true',
        'STRICT_MODE_ENABLED': 'true',
        'ENABLE_MULTIMODAL_FUSION': 'true',
        'ENABLE_FORENSIC_ANALYSIS': 'true',
        'ENABLE_MODEL_EXPLAINABILITY': 'true'
    }
    
    # Files to update
    env_files = [
        Path('.env'),
        Path('.env.example'),
        Path('.env.production'),
        Path('server/.env'),
        Path('server/python/.env')
    ]
    
    # Update all environment files
    for env_file in env_files:
        if env_file.exists():
            print(f"\n📝 Updating {env_file}")
            for var_name, value in ml_vars.items():
                set_env_var(env_file, var_name, value)
    
    # Update Python scripts
    print(f"\n🐍 Updating Python configuration files...")
    
    python_files = [
        Path('server/python/main_api.py'),
        Path('server/python/config_strict_mode.py')
    ]
    
    for py_file in python_files:
        if py_file.exists():
            content = py_file.read_text()
            # Force enable ML models in Python code
            if 'ENABLE_ML_MODELS' in content:
                content = content.replace(
                    "ENABLE_ML_MODELS = os.getenv('ENABLE_ML_MODELS', 'false').lower() == 'true'",
                    "ENABLE_ML_MODELS = True"
                )
                content = content.replace(
                    "ENABLE_ADVANCED_MODELS = os.getenv('ENABLE_ADVANCED_MODELS', 'false').lower() == 'true'",
                    "ENABLE_ADVANCED_MODELS = True"
                )
                py_file.write_text(content)
                print(f"✅ Updated {py_file}")
    
    # Update frontend configuration
    print(f"\n⚛️ Updating frontend configuration...")
    vite_config = Path('client/vite.config.ts')
    if vite_config.exists():
        content = vite_config.read_text()
        if 'define:' in content:
            # Add ML model environment variables to Vite config
            ml_define = """
      // Force enable ML models in frontend
      'process.env.ENABLE_ML_MODELS': JSON.stringify('true'),
      'process.env.ENABLE_ADVANCED_MODELS': JSON.stringify('true'),
      'process.env.STRICT_MODE_ENABLED': JSON.stringify('true'),"""
            
            if 'process.env.ENABLE_ML_MODELS' not in content:
                content = content.replace(
                    "define: {",
                    f"define: {{{ml_define}"
                )
                vite_config.write_text(content)
                print(f"✅ Updated {vite_config}")
    
    print("\n" + "=" * 60)
    print("🎯 ML Models Enabled Everywhere!")
    print("📋 Summary:")
    print("   ✅ Environment variables updated")
    print("   ✅ Python scripts configured")
    print("   ✅ Frontend configuration updated")
    print("   ✅ All ML models will be loaded on startup")
    print("\n🔄 Restart all services to apply changes:")
    print("   npm run start:all")
    print("=" * 60)

if __name__ == "__main__":
    main()
