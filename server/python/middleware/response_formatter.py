"""
Response Formatter Middleware
Standardizes API responses and error handling.
"""
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standard error response format"""

    success: bool = False
    error: str
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Standard success response format"""

    success: bool = True
    data: Any
    meta: Optional[Dict[str, Any]] = None


def format_error(
    status_code: int,
    error: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Format an error response."""
    error_codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_server_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }

    error_code = error_codes.get(status_code, "internal_error")

    response = ErrorResponse(
        error=error_code,
        code=error_code.upper(),
        message=message,
        details=details or {},
    )

    return JSONResponse(
        status_code=status_code, content=response.dict(), headers=headers or {}
    )


def format_success(
    data: Any = None,
    meta: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Format a success response."""
    response = SuccessResponse(data=data, meta=meta or {})

    return JSONResponse(
        status_code=status_code, content=response.dict(), headers=headers or {}
    )


class ResponseFormatterMiddleware(BaseHTTPMiddleware):
    """Middleware to format all responses consistently."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        try:
            # Process the request
            response = await call_next(request)

            # Skip formatting for certain paths
            if self._should_skip_formatting(request, response):
                return response

            # Process successful responses
            if 200 <= response.status_code < 300:
                content = await self._get_response_content(response)
                return format_success(
                    data=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            # Process error responses
            content = await self._get_response_content(response)
            error_info = self._extract_error_info(response.status_code, content)

            return format_error(
                status_code=response.status_code,
                error=error_info["error"],
                message=error_info["message"],
                details=error_info.get("details", {}),
                headers=dict(response.headers),
            )

        except Exception as e:
            # Handle uncaught exceptions
            logger.exception("Unhandled exception in request")
            return format_error(
                status_code=500,
                error="internal_server_error",
                message="An unexpected error occurred",
                details={"exception": str(e) if str(e) else "Unknown error"},
            )
        finally:
            # Log request completion
            process_time = (time.time() - start_time) * 1000
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"completed in {process_time:.2f}ms"
            )

    def _should_skip_formatting(self, request: Request, response: Response) -> bool:
        """Check if response should skip formatting."""
        skip_paths = {"/docs", "/redoc", "/openapi.json", "/static", "/health"}

        return any(request.url.path.startswith(path) for path in skip_paths)

    async def _get_response_content(self, response: Response) -> Any:
        """Extract content from response."""
        try:
            if hasattr(response, "body"):
                content = response.body
                if hasattr(content, "decode"):
                    content = content.decode()
                if content:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
        except Exception:
            logger.warning("Failed to parse response content")
        return None

    def _extract_error_info(self, status_code: int, content: Any) -> Dict[str, Any]:
        """Extract error information from response content."""
        if isinstance(content, dict):
            return {
                "error": content.get("error", "unknown_error"),
                "message": content.get("message", "An error occurred"),
                "details": content.get("details", {}),
            }

        return {
            "error": "request_error",
            "message": str(content) if content else "An error occurred",
            "details": {},
        }
