"""
Cache Manager Service
Redis-based caching for analysis results and sessions
"""

import json
import logging
import os
from typing import Optional, Any, Union
import redis
from datetime import timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Cache manager using Redis
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", host),
                port=int(os.getenv("REDIS_PORT", port)),
                db=int(os.getenv("REDIS_DB", db)),
                password=os.getenv("REDIS_PASSWORD", password),
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis.ping()
            self.enabled = True
            logger.info("✅ Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache not available: {e}")
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            value = self.redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: Union[int, timedelta] = 3600) -> bool:
        """Set value in cache"""
        if not self.enabled:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            return self.redis.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.enabled:
            return False
        
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.enabled:
            return False
        
        try:
            return bool(self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self.enabled:
            return 0
        
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
