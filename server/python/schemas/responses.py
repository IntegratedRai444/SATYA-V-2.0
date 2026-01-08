"""
Standardized response models for the API
"""
from typing import Generic, TypeVar, Optional, Any, Dict, List
from pydantic import BaseModel, Field

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    """Base response model with success status and optional data"""
    success: bool = Field(..., description="Indicates if the request was successful")
    data: Optional[T] = Field(None, description="Response data if successful")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if request failed")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")

class PaginatedResponse(BaseResponse[T]):
    """Response model for paginated data"""
    total: int = Field(..., description="Total number of items available")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    items: List[T] = Field(default_factory=list, description="List of items in the current page")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server timestamp")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Status of service dependencies")
