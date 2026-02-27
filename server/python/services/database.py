"""
Database Service for Python ML Service
Handles database operations and connection management
"""

import logging
import os
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager for Python service"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/satya')
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Create async engine
            self.async_engine = create_async_engine(
                self.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
                echo=False,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Test connection
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                
            logger.info("✅ Database service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database service initialization failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as health"))
                row = result.fetchone()
                return {
                    "healthy": True,
                    "database_connected": True,
                    "test_query": row[0] if row else None
                }
        except Exception as e:
            return {
                "healthy": False,
                "database_connected": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Database connections closed")

# Global instance
database_manager = DatabaseManager()
