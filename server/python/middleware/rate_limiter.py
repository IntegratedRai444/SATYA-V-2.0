import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import redis
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

from config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enhanced rate limiting middleware with Redis support for distributed environments.

    Features:
    - Per-IP rate limiting
    - Per-route rate limiting
    - Distributed rate limiting using Redis
    - Request fingerprinting for more accurate rate limiting
    - Support for different rate limits per HTTP method
    """

    def __init__(
        self,
        app,
        redis_url: Optional[str] = None,
        default_rate_limit: str = "100/minute",
        rate_limits: Optional[Dict[str, str]] = None,
        block_duration: int = 300,
        enabled: bool = True,
        redis_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(app)
        self.enabled = enabled
        self.block_duration = block_duration
        self.rate_limits = rate_limits or {}
        self.default_limit, self.default_period = self._parse_rate_limit(
            default_rate_limit
        )
        self.redis_options = redis_options or {}

        # Initialize Redis client if URL is provided
        self.redis = None
        if redis_url:
            try:
                self.redis = redis.Redis.from_url(
                    redis_url, decode_responses=True, **self.redis_options
                )
                # Test connection
                self.redis.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis = None

    async def dispatch(
        self, request: StarletteRequest, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for certain paths
        path = request.url.path
        if any(
            path.startswith(p)
            for p in [
                "/health",
                "/metrics",
                "/api/v2/health",
                "/api/v2/docs",
                "/api/v2/redoc",
                "/api/v2/openapi.json",
            ]
        ):
            return await call_next(request)

        # Get client identifier (IP + user agent for better accuracy)
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        client_id = self._generate_fingerprint(
            client_ip, user_agent, path, request.method
        )

        # Get rate limit for this endpoint or use default
        rate_limit = self._get_rate_limit_for_path(path, request.method)
        limit, period = self._parse_rate_limit(rate_limit)

        # Check rate limit
        is_blocked, retry_after = await self._check_rate_limit(
            client_id=client_id, limit=limit, period=period
        )

        if is_blocked:
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "code": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                },
            )
            response.headers["Retry-After"] = str(retry_after)
            return response

        # Process the request
        response = await call_next(request)

        # Add rate limit headers
        remaining, reset = await self._get_rate_limit_info(client_id, limit, period)

        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)

        return response

    def _get_client_ip(self, request: StarletteRequest) -> str:
        """Get client IP address from request headers"""
        # Check for X-Forwarded-For header (common with proxies)
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()

        # Check for X-Real-IP header (common with nginx)
        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip

        # Fall back to client host (may be a proxy)
        if request.client and request.client.host:
            return request.client.host

        return "unknown"

    def _generate_fingerprint(
        self, ip: str, user_agent: str, path: str, method: str
    ) -> str:
        """Generate a unique fingerprint for rate limiting"""
        key = f"{ip}:{user_agent}:{path}:{method}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_rate_limit_for_path(self, path: str, method: str) -> str:
        """Get rate limit for a specific path and method"""
        # Check for exact path match
        if path in self.rate_limits:
            return self.rate_limits[path]

        # Check for method-specific rate limits
        method_limit = f"{method}:{path}"
        if method_limit in self.rate_limits:
            return self.rate_limits[method_limit]

        # Check for prefix matches (e.g., "/api/")
        for prefix, limit in self.rate_limits.items():
            if path.startswith(prefix):
                return limit

        # Default rate limit
        return f"{self.default_limit}/{self.default_period}second"

    async def _check_rate_limit(
        self, client_id: str, limit: int, period: int
    ) -> Tuple[bool, int]:
        """Check if the request exceeds the rate limit"""
        current_time = int(time.time())
        window_start = current_time - period

        if self.redis:
            # Use Redis for distributed rate limiting
            try:
                # Use a Redis pipeline for atomic operations
                with self.redis.pipeline() as pipe:
                    # Add current timestamp to the sorted set
                    pipe.zadd(
                        f"ratelimit:{client_id}", {str(current_time): current_time}
                    )
                    # Remove old timestamps
                    pipe.zremrangebyscore(
                        f"ratelimit:{client_id}", "-inf", window_start
                    )
                    # Get count of requests in current window
                    pipe.zcard(f"ratelimit:{client_id}")
                    # Set TTL on the key
                    pipe.expire(f"ratelimit:{client_id}", period)
                    _, _, count, _ = pipe.execute()

                # Check if limit is exceeded
                if count > limit:
                    # Get the oldest timestamp to calculate retry_after
                    oldest = self.redis.zrange(
                        f"ratelimit:{client_id}", 0, 0, withscores=True
                    )
                    if oldest:
                        retry_after = int((oldest[0][1] + period) - current_time)
                        return True, max(1, retry_after)
                    return True, period

                return False, 0

            except redis.RedisError as e:
                logger.error(f"Redis error in rate limiting: {e}")
                # Fail open - allow the request if Redis is down
                return False, 0
        else:
            # In-memory rate limiting (fallback)
            if not hasattr(self, "_request_timestamps"):
                self._request_timestamps = {}

            if client_id not in self._request_timestamps:
                self._request_timestamps[client_id] = []

            # Remove timestamps outside the current window
            self._request_timestamps[client_id] = [
                ts for ts in self._request_timestamps[client_id] if ts > window_start
            ]

            # Check if limit is exceeded
            if len(self._request_timestamps[client_id]) >= limit:
                # Calculate retry_after based on oldest timestamp
                retry_after = int(
                    (self._request_timestamps[client_id][0] + period) - current_time
                )
                return True, max(1, retry_after)

            # Add current timestamp
            self._request_timestamps[client_id].append(current_time)
            return False, 0

    async def _get_rate_limit_info(
        self, client_id: str, limit: int, period: int
    ) -> Tuple[int, int]:
        """Get remaining requests and reset time"""
        current_time = int(time.time())
        window_start = current_time - period

        if self.redis:
            try:
                # Get count of requests in current window
                count = self.redis.zcount(
                    f"ratelimit:{client_id}", window_start, "+inf"
                )
                # Get oldest timestamp to calculate reset time
                oldest = self.redis.zrange(
                    f"ratelimit:{client_id}", 0, 0, withscores=True
                )
                reset_at = (
                    (oldest[0][1] + period) if oldest else (current_time + period)
                )
                return max(0, limit - count), int(reset_at)
            except redis.RedisError as e:
                logger.error(f"Redis error getting rate limit info: {e}")
                return limit, current_time + period
        else:
            # In-memory fallback
            if (
                not hasattr(self, "_request_timestamps")
                or client_id not in self._request_timestamps
            ):
                return limit, current_time + period

            count = len(
                [ts for ts in self._request_timestamps[client_id] if ts > window_start]
            )

            if not self._request_timestamps[client_id]:
                return limit, current_time + period

            reset_at = self._request_timestamps[client_id][0] + period
            return max(0, limit - count), int(reset_at)

    def _parse_rate_limit(self, rate_limit: str) -> Tuple[int, int]:
        """Parse rate limit string (e.g., '100/minute' -> (100, 60))"""
        try:
            limit, period = rate_limit.split("/")
            limit = int(limit)

            if period.startswith("second"):
                period_seconds = 1
            elif period.startswith("minute"):
                period_seconds = 60
            elif period.startswith("hour"):
                period_seconds = 3600
            elif period.startswith("day"):
                period_seconds = 86400
            else:
                # Default to seconds if period is not recognized
                period_seconds = int(period) if period.isdigit() else 60

            return limit, period_seconds

        except Exception as e:
            logger.warning(f"Invalid rate limit format: {rate_limit}. Using default.")
            return self.default_limit, self.default_period


# Simple RateLimiter class for basic rate limiting
class RateLimiter:
    """Simple rate limiter for basic functionality."""
    def __init__(self, requests=100, window=60):
        self.requests = requests
        self.window = window
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, key=None):
        """Simple check - always allow for now (Redis would be needed for real rate limiting)."""
        return True

# Global rate limiter instance
rate_limiter = RateLimitMiddleware(
    app=None,
    redis_url=settings.REDIS_URL,
    default_rate_limit="100/minute",
    rate_limits={"/api/v2/upload": "5/minute", "/api/v2/retrieve": "60/minute"}
)

# Global rate limiters (created when module is loaded)
AUDIO_UPLOAD_LIMITER = None
AUDIO_RETRIEVE_LIMITER = None

# Initialize rate limiters when module is loaded
def _initialize_rate_limiters():
    global AUDIO_UPLOAD_LIMITER, AUDIO_RETRIEVE_LIMITER
    try:
        AUDIO_UPLOAD_LIMITER = RateLimiter(requests=5, window=60)  # 5 uploads per minute
        AUDIO_RETRIEVE_LIMITER = RateLimiter(requests=60, window=60)  # 60 retrieves per minute
    except Exception as e:
        logger.error(f"Failed to initialize rate limiters: {e}")
        AUDIO_UPLOAD_LIMITER = None
        AUDIO_RETRIEVE_LIMITER = None

# Initialize on import
_initialize_rate_limiters()

# Backward compatibility alias
RateLimiterMiddleware = RateLimitMiddleware
