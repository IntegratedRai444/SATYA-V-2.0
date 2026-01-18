"""
Database Base Configuration
Supabase client setup and configuration
"""

import os
from typing import Generator, Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase() -> Client:
    """
    Dependency to get Supabase client
    """
    return supabase
