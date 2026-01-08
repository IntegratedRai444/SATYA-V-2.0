"""
Request validation schemas using Pydantic
"""
import re
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# Allowed file extensions for each media type
ALLOWED_EXTENSIONS = {
    MediaType.IMAGE: [".jpg", ".jpeg", ".png", ".webp"],
    MediaType.VIDEO: [".mp4", ".avi", ".mov", ".mkv"],
    MediaType.AUDIO: [".mp3", ".wav", ".ogg", ".flac"],
}

# Maximum file sizes in bytes (10MB for images, 100MB for videos/audio)
MAX_FILE_SIZES = {
    MediaType.IMAGE: 10 * 1024 * 1024,  # 10MB
    MediaType.VIDEO: 100 * 1024 * 1024,  # 100MB
    MediaType.AUDIO: 50 * 1024 * 1024,  # 50MB
}


class BaseRequest(BaseModel):
    """Base request model with common fields"""

    metadata: Optional[dict] = Field(
        default_factory=dict, description="Additional metadata for the request"
    )


class FileUploadRequest(BaseRequest):
    """Model for file upload requests"""

    file: bytes = Field(..., description="Binary file data")
    filename: str = Field(
        ..., description="Original filename with extension", example="example.jpg"
    )
    media_type: MediaType = Field(..., description="Type of media being uploaded")

    @validator("filename")
    def validate_filename(cls, v, values):
        """Validate file extension"""
        if "media_type" not in values:
            return v

        ext = "." + v.rsplit(".", 1)[-1].lower()
        allowed_exts = ALLOWED_EXTENSIONS[values["media_type"]]

        if ext not in allowed_exts:
            raise ValueError(
                f"Invalid file extension for {values['media_type']}. "
                f"Allowed extensions: {', '.join(allowed_extensions)}"
            )
        return v

    @validator("file")
    def validate_file_size(cls, v, values):
        """Validate file size"""
        if "media_type" not in values:
            return v

        max_size = MAX_FILE_SIZES[values["media_type"]]
        if len(v) > max_size:
            raise ValueError(
                f"File too large. Maximum size for {values['media_type']} "
                f"is {max_size // (1024 * 1024)}MB"
            )
        return v


class URLRequest(BaseRequest):
    """Model for URL-based requests"""

    url: HttpUrl = Field(..., description="URL of the media to analyze")
    media_type: MediaType = Field(..., description="Type of media being analyzed")


class BatchRequest(BaseRequest):
    """Model for batch processing requests"""

    items: List[Union[FileUploadRequest, URLRequest]] = Field(
        ...,
        max_items=10,  # Limit batch size to 10 items
        description="List of files or URLs to process",
    )


class AnalysisConfig(BaseModel):
    """Configuration for analysis"""

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection (0.0 to 1.0)",
    )
    include_heatmap: bool = Field(
        default=False, description="Whether to include heatmap in the response"
    )
    return_features: bool = Field(
        default=False, description="Whether to include feature vectors in the response"
    )


# Response models
class DetectionResult(BaseModel):
    """Model for detection results"""

    is_fake: bool
    confidence: float
    metadata: dict = {}
    analysis_time: float
    model_version: str


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str
    details: Optional[dict] = None
