"""
Cache Manager Service
Redis-based caching for analysis results and sessions
"""

import json
import logging
import os
from datetime import timedelta
from typing import Any, Optional, Union

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache manager using Redis
    """

    def __init__(
        self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 3600
    ):
        """Initialize cache manager"""
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client = None
        
        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis not available - caching disabled")
            return
            
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("✅ Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache not available: {e}")
            self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not hasattr(self, 'enabled') or not self.enabled:
            return None

        try:
            value = self.redis_client.get(key)
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
        if not hasattr(self, 'enabled') or not self.enabled:
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)

            return self.redis_client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not hasattr(self, 'enabled') or not self.enabled:
            return False

        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not hasattr(self, 'enabled') or not self.enabled:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not hasattr(self, 'enabled') or not self.enabled:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
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
