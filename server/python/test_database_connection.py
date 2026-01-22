#!/usr/bin/env python3
"""
Database Connection Test Script
Tests the enhanced database connection functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from services.database import get_db_manager
from database.base import get_supabase

async def test_database_connection():
    """Test database connection with enhanced error handling"""
    
    print("🔍 Testing Database Connection...")
    print("=" * 50)
    
    try:
        # Test 1: Basic Supabase client initialization
        print("1. Testing Supabase client initialization...")
        supabase = get_supabase()
        print("✅ Supabase client initialized successfully")
        
        # Test 2: Database manager initialization
        print("\n2. Testing Database Manager initialization...")
        db_manager = get_db_manager()
        print("✅ Database Manager initialized successfully")
        
        # Test 3: Connection test
        print("\n3. Testing database connection...")
        connection_result = await db_manager.test_connection()
        if connection_result:
            print("✅ Database connection test PASSED")
        else:
            print("❌ Database connection test FAILED")
            return False
        
        # Test 4: Health check
        print("\n4. Testing database health check...")
        health_status = await db_manager.health_check()
        print(f"📊 Database Health: {health_status}")
        
        # Test 5: Is connected check
        print("\n5. Testing is_connected method...")
        is_connected = db_manager.is_connected()
        print(f"🔌 Is Connected: {is_connected}")
        
        # Test 6: Simple query test
        print("\n6. Testing simple database query...")
        try:
            result = supabase.table("users").select("count").limit(1).execute()
            if result is not None:
                print("✅ Simple query test PASSED")
            else:
                print("❌ Simple query test FAILED - No response")
        except Exception as e:
            print(f"❌ Simple query test FAILED: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 All database tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database test FAILED: {e}")
        print("=" * 50)
        return False

async def main():
    """Main test function"""
    print("SatyaAI Python Backend - Database Connection Test")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    
    print(f"✅ Environment variables OK")
    print(f"📡 Supabase URL: {os.getenv('SUPABASE_URL')}")
    
    # Run database tests
    success = await test_database_connection()
    
    if success:
        print("\n🚀 Database connection is ready for production!")
    else:
        print("\n⚠️ Database connection needs attention before production use")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
