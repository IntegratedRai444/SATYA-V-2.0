"""
Database Configuration for Supabase
Handles Supabase client initialization and database operations
"""
import os
from typing import Generator, Optional, Dict, Any, List
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase() -> Client:
    """
    Dependency to get Supabase client
    
    Usage in FastAPI:
        @app.get("/items/")
        def read_items(supabase: Client = Depends(get_supabase)):
            data = supabase.table('items').select('*').execute()
            return data.data
    """
    return supabase

def execute_query(
    table: str, 
    query_type: str = "select", 
    filters: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    columns: str = "*"
) -> List[Dict[str, Any]]:
    """
    Execute a query on Supabase
    
    Args:
        table: Name of the table to query
        query_type: Type of query (select, insert, update, delete)
        filters: Dictionary of filters for the query
        data: Data for insert/update operations
        columns: Columns to select (default: "*")
    """
    try:
        query = supabase.table(table)
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    query = query.in_(key, value)
                else:
                    query = query.eq(key, value)
        
        if query_type == "select":
            result = query.select(columns).execute()
            return result.data if hasattr(result, 'data') else []
            
        elif query_type == "insert" and data:
            result = query.insert(data).execute()
            return result.data if hasattr(result, 'data') else []
            
        elif query_type == "update" and data:
            result = query.update(data).execute()
            return result.data if hasattr(result, 'data') else []
            
        elif query_type == "delete":
            result = query.delete().execute()
            return result.data if hasattr(result, 'data') else []
            
        return []
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
