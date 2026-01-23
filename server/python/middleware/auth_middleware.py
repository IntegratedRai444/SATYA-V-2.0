"""
Authentication Middleware
Verifies Supabase JWT tokens from Node.js gateway.
"""
import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.requests import Request
from starlette.responses import Response
import httpx

# Configure logging
logger = logging.getLogger(__name__)


class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        supabase_url: str,
        supabase_anon_key: str,
        public_paths: List[str] = None,
    ):
        super().__init__(app)
        self.supabase_url = supabase_url
        self.supabase_anon_key = supabase_anon_key
        self.supabase_client = httpx.AsyncClient(
            base_url=f"{supabase_url}/auth/v1",
            headers={"Authorization": f"Bearer {supabase_anon_key}"}
        )

        # Define default public paths (can be overridden)
        self.public_paths = set(public_paths) if public_paths else set()

        # Add default public paths
        self.public_paths.update(
            {
                "/api/v2/docs",
                "/api/v2/redoc",
                "/api/v2/openapi.json",
                "/api/v2/health",
                "/health",
                "/",
            }
        )

        # Add regex patterns for dynamic public paths (e.g., /static/*)
        self.public_patterns = [
            re.compile(r"^/static/.*$"),
            re.compile(r"^/media/.*$"),
            re.compile(r"^/healthz$"),
            re.compile(r"^/favicon\.ico$"),
        ]

        # Cache for path checks
        self._path_cache: Dict[str, bool] = {}

    def is_public_path(self, path: str) -> bool:
        """Check if a given path is public."""
        # Check cache first
        if path in self._path_cache:
            return self._path_cache[path]

        # Check exact matches
        if path in self.public_paths:
            self._path_cache[path] = True
            return True

        # Check path prefixes
        for public_path in self.public_paths:
            if path.startswith(public_path):
                self._path_cache[path] = True
                return True

        # Check regex patterns
        for pattern in self.public_patterns:
            if pattern.match(path):
                self._path_cache[path] = True
                return True

        # Not a public path
        self._path_cache[path] = False
        return False

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip authentication for public paths
        if self.is_public_path(request.url.path):
            return await call_next(request)

        # Get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning(
                f"No valid Authorization header for protected path: {request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ")[1]

        # Verify token with Supabase
        try:
            response = await self.supabase_client.get(
                "/user",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Invalid Supabase token: {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user_data = response.json()
            
            # Add user info to request state
            request.state.user = {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "email_confirmed_at": user_data.get("email_confirmed_at"),
                "user_metadata": user_data.get("user_metadata", {}),
                "is_active": True,
                "is_verified": user_data.get("email_confirmed_at") is not None,
            }

        except httpx.HTTPError as e:
            logger.error(f"Supabase verification error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable",
            )
        except Exception as e:
            logger.error(f"Unexpected authentication error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authentication",
            )

        # Continue with the request
        return await call_next(request)


def auth_required(permissions: List[str] = None, require_verified: bool = True):
    """
    Decorator to protect routes with JWT authentication and optional permission checks.

    Args:
        permissions: List of required permissions (if any)
        require_verified: Whether the user's email must be verified
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request.state, "user") or not request.state.user:
                logger.warning("Unauthenticated access attempt to protected endpoint")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user = request.state.user

            # Check if email verification is required
            if require_verified and not user.get("is_verified", False):
                logger.warning(f"Unverified user attempted access: {user.get('email')}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Email verification required",
                )

            # Check permissions if required
            if permissions:
                user_permissions = set(user.get("permissions", []))
                required_permissions = set(permissions)

                if not required_permissions.issubset(user_permissions):
                    logger.warning(
                        f"Insufficient permissions for user {user.get('email')}. "
                        f"Required: {required_permissions}, Has: {user_permissions}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions",
                    )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


# Helper function to get current user from request state
async def get_current_user(request: Request) -> Dict[str, Any]:
    if not hasattr(request.state, "user") or not request.state.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return request.state.user


# Dependency for FastAPI endpoints
CurrentUser = Depends(get_current_user)


def public_endpoint(func):
    """Decorator to mark an endpoint as public (no auth required)."""
    func.is_public = True
    return func


def check_auth_middleware():
    """Factory function to create auth middleware."""
    security = HTTPBearer()

    async def middleware(request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in security.public_paths or request.url.path.startswith(
            "/static"
        ):
            return await call_next(request)

        # Check for authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify token
        try:
            token = auth_header.split(" ")[1]  # Bearer <token>
            payload = jwt.decode(
                token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM]
            )
            request.state.user_id = payload.get("sub")
            return await call_next(request)
        except (JWTError, IndexError) as e:
            logger.warning(f"JWT validation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return middleware
