"""
Authentication Middleware
Enforces authentication on all routes by default.
Public routes must be explicitly marked with @public_endpoint decorator.
"""
import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import (HTTPAuthorizationCredentials, HTTPBearer,
                              OAuth2PasswordBearer)
from jose import JWTError, jwt
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.requests import Request
from starlette.responses import Response

# Configure logging
logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        secret_key: str,
        algorithm: str = "HS256",
        public_paths: List[str] = None,
    ):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer(auto_error=False)

        # Define default public paths (can be overridden)
        self.public_paths = set(public_paths) if public_paths else set()

        # Add default public paths
        self.public_paths.update(
            {
                "/api/v2/docs",
                "/api/v2/redoc",
                "/api/v2/openapi.json",
                "/api/v2/health",
                "/api/v2/auth/login",
                "/api/v2/auth/register",
                "/api/v2/auth/refresh",
                "/api/v2/auth/verify-email",
                "/api/v2/auth/forgot-password",
                "/api/v2/auth/reset-password",
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
        """Check if the given path is public."""
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

        # Get the token from the Authorization header
        credentials: Optional[HTTPAuthorizationCredentials] = None
        try:
            credentials = await self.security(request)
        except Exception as e:
            logger.warning(f"Error extracting credentials: {str(e)}")

        # If no credentials provided, reject the request
        if not credentials or not credentials.credentials:
            logger.warning(
                f"No credentials provided for protected path: {request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials

        # Verify the token
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_aud": False},
            )

            # Add user info to request state
            request.state.user = {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "permissions": payload.get("permissions", []),
                "is_active": payload.get("is_active", True),
                "is_verified": payload.get("is_verified", False),
            }

            # Check if user is active
            if not request.state.user.get("is_active", False):
                logger.warning(f"Inactive user attempted access: {request.state.user}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive",
                )

        except JWTError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
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
