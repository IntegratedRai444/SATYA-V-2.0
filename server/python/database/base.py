

import os
import logging
from typing import Generator, Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client with enhanced validation
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Enhanced validation with clear error messages
if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL must be set in environment variables. "
        "Please add it to your .env file or set it in your environment."
    )

if not SUPABASE_ANON_KEY and not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Neither SUPABASE_ANON_KEY nor SUPABASE_SERVICE_ROLE_KEY is set. "
        "Please add one of these to your .env file or set it in your environment."
    )

# Try service role key first (more permissions), then anon key
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY

# Log which key we're using (without exposing the actual key)
key_type = "service_role" if SUPABASE_KEY == SUPABASE_SERVICE_ROLE_KEY else "anon"
logger = logging.getLogger(__name__)
logger.info(f"🔑 Using Supabase {key_type} key for database connection")

# Create Supabase client with error handling
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase client: {e}")
    raise ValueError(
        f"Failed to initialize Supabase client: {e}. "
        "Please check your SUPABASE_URL and SUPABASE_KEY configuration."
    )

def get_supabase() -> Client:
    
    try:
        return supabase
    except Exception as e:
        logger.error(f"❌ Failed to get Supabase client: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable. Please check configuration."
        )

def validate_supabase_connection() -> Dict[str, Any]:
    
    try:
       
        result = supabase.table("users").select("count").limit(1).execute()
        
        return {
            "status": "connected",
            "message": "Supabase connection successful",
            "key_type": key_type,
            "url_configured": bool(SUPABASE_URL),
            "key_configured": bool(SUPABASE_KEY)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Supabase connection failed: {str(e)}",
            "key_type": key_type,
            "url_configured": bool(SUPABASE_URL),
            "key_configured": bool(SUPABASE_KEY),
            "error": str(e)
        }
