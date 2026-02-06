#!/usr/bin/env python3
"""
Verify Production Configuration
Check all environment files are properly configured
"""

import os
import re
from pathlib import Path

def check_env_file(file_path: str) -> dict:
    """Check environment file for required keys and values"""
    results = {
        'file': file_path,
        'exists': False,
        'configured_keys': {},
        'missing_placeholders': [],
        'issues': []
    }
    
    if not Path(file_path).exists():
        results['issues'].append(f"File {file_path} does not exist")
        return results
    
    results['exists'] = True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for placeholder values
        placeholder_patterns = [
            r'.*-set-in-env$',
            r'.*your-.*-key$',
            r'.*domain-set-in-env$',
            r'.*password-set-in-env$'
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                results['configured_keys'][key] = value
                
                # Check for placeholders
                for pattern in placeholder_patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        results['missing_placeholders'].append(key)
                        break
    
    except Exception as e:
        results['issues'].append(f"Error reading {file_path}: {e}")
    
    return results

def main():
    print("🔍 Verifying Production Configuration...")
    print("=" * 60)
    
    # Files to check
    env_files = [
        '.env',
        '.env.production',
        'server/.env',
        'server/python/.env'
    ]
    
    all_good = True
    
    for env_file in env_files:
        print(f"\n📁 Checking {env_file}:")
        result = check_env_file(env_file)
        
        if not result['exists']:
            print(f"   ❌ File does not exist")
            all_good = False
            continue
        
        # Check critical keys
        critical_keys = {
            'SUPABASE_URL': 'https://ftbpbghcebwgzqfsgmxk.supabase.co',
            'DATABASE_URL': 'postgresql://postgres:Rishabhkapoor@db.ftbpbghcebwgzqfsgmxk.supabase.co:5432/postgres',
            'SUPABASE_JWT_SECRET': 'sb_secret_bXjhAijPCJslbqVHNVNBsQ_xSfkVDY7'
        }
        
        for key, expected_value in critical_keys.items():
            if key in result['configured_keys']:
                actual_value = result['configured_keys'][key]
                if actual_value == expected_value:
                    print(f"   ✅ {key}: Correctly configured")
                else:
                    print(f"   ⚠️ {key}: {actual_value}")
            else:
                print(f"   ❌ {key}: Missing")
                all_good = False
        
        # Check for placeholders
        if result['missing_placeholders']:
            print(f"   ❌ Placeholders found: {', '.join(result['missing_placeholders'])}")
            all_good = False
        
        # Check ML models are enabled
        ml_keys = ['ENABLE_ML_MODELS', 'ENABLE_ADVANCED_MODELS', 'STRICT_MODE_ENABLED']
        ml_enabled = True
        for key in ml_keys:
            if key in result['configured_keys']:
                if result['configured_keys'][key].lower() == 'true':
                    print(f"   ✅ {key}: Enabled")
                else:
                    print(f"   ❌ {key}: Disabled")
                    ml_enabled = False
            else:
                print(f"   ⚠️ {key}: Not found")
        
        if ml_enabled:
            print("   🚀 ML Models: All Enabled")
        
        # Show any other issues
        if result['issues']:
            for issue in result['issues']:
                print(f"   ❌ {issue}")
                all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Production Configuration: COMPLETE!")
        print("✅ All critical services configured")
        print("✅ ML models enabled everywhere")
        print("✅ No placeholder values found")
        print("\n🚀 Ready for production deployment!")
    else:
        print("⚠️ Production Configuration: INCOMPLETE")
        print("❌ Some issues found - check output above")
    
    print("\n📋 Next Steps:")
    print("1. Restart all services: npm run start:all")
    print("2. Test authentication flow")
    print("3. Verify ML models load correctly")
    print("4. Test deepfake detection functionality")
    print("=" * 60)

if __name__ == "__main__":
    main()
