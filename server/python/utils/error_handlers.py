"""
Standardized error handling for the API
"""
from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ErrorResponse(BaseModel):
    error: str
    code: int
    details: Optional[Dict[str, Any]] = None

def http_exception(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> HTTPException:
    """Create a standardized HTTP exception"""
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            error=message,
            code=status_code,
            details=details or {},
        ).model_dump(),
        headers=headers or {},
    )

# Common error responses
class Errors:
    NOT_FOUND = http_exception(
        status_code=status.HTTP_404_NOT_FOUND,
        message="Resource not found",
    )
    
    UNAUTHORIZED = http_exception(
        status_code=status.HTTP_401_UNAUTHORIZED,
        message="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    FORBIDDEN = http_exception(
        status_code=status.HTTP_403_FORBIDDEN,
        message="Insufficient permissions",
    )
    
    BAD_REQUEST = http_exception(
        status_code=status.HTTP_400_BAD_REQUEST,
        message="Bad request",
    )
    
    INTERNAL_ERROR = http_exception(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="Internal server error",
    )
    
    SERVICE_UNAVAILABLE = http_exception(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        message="Service temporarily unavailable",
    )
    
    @staticmethod
    def validation_error(details: Dict[str, Any]) -> HTTPException:
        return http_exception(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation error",
            details={"fields": details},
        )
