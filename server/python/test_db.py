#!/usr/bin/env python3
"""
Quick database connection test
"""
import os
import sys
sys.path.insert(0, '.')

try:
    from services.database import get_db_manager
    print("✅ Database service imported successfully")
except ImportError as e:
    print(f"❌ Failed to import database service: {e}")
    sys.exit(1)

async def test_database():
    """Test database connection"""
    print("🔍 Testing database connection...")
    
    try:
        # Check environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        print(f"📋 Environment check:")
        print(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Missing'}")
        print(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Missing'}")
        
        if not supabase_url or not supabase_key:
            print("❌ Missing required environment variables")
            return False
        
        # Test database manager
        db_manager = get_db_manager()
        print("✅ Database manager created")
        
        # Test connection
        result = await db_manager.test_connection()
        print(f"🔌 Connection test result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_database())
    
    if result:
        print("🎉 Database connection successful!")
        sys.exit(0)
    else:
        print("💥 Database connection failed!")
        sys.exit(1)
