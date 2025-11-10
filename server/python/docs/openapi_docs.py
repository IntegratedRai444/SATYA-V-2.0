"""
SatyaAI OpenAPI Documentation

This module provides OpenAPI (Swagger) documentation for the SatyaAI API.
It includes detailed documentation for all API endpoints, request/response schemas,
and example requests.
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# Initialize FastAPI app for documentation
app = FastAPI(
    title="SatyaAI API",
    description="""
    ## Professional Deepfake Detection API
    
    SatyaAI provides state-of-the-art deepfake detection for images, videos, and audio.
    This API allows you to analyze media files for potential deepfake content using
    multiple AI models and techniques.
    
    ### Authentication
    Most endpoints require authentication. Include your API key in the `Authorization` header:
    ```
    Authorization: Bearer YOUR_API_KEY
    ```
    """,
    version="2.0.0",
    contact={
        "name": "SatyaAI Support",
        "email": "support@satyaai.tech"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "Image Analysis",
            "description": "Deepfake detection for images"
        },
        {
            "name": "Video Analysis",
            "description": "Deepfake detection for videos"
        },
        {
            "name": "Audio Analysis",
            "description": "Deepfake detection for audio files"
        },
        {
            "name": "System",
            "description": "System status and health checks"
        }
    ]
)

# Models for request/response schemas
class DetectionResult(BaseModel):
    """Result of a deepfake detection analysis"""
    is_real: bool = Field(..., description="Whether the media is determined to be real")
    confidence: float = Field(..., description="Confidence score (0-1) of the detection")
    model_used: str = Field(..., description="Name of the model used for detection")
    processing_time: float = Field(..., description="Time taken for analysis in seconds")
    details: dict = Field(..., description="Detailed analysis results")

class AnalysisRequest(BaseModel):
    """Request model for media analysis"""
    file_url: Optional[str] = Field(None, description="URL of the file to analyze")
    model_preference: Optional[str] = Field(None, description="Preferred model to use for analysis")
    detailed_report: bool = Field(False, description="Whether to include detailed analysis")

class UserCreate(BaseModel):
    """User registration model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    """User response model"""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

# Example endpoints with documentation
@app.post("/api/v1/analyze/image", 
          response_model=DetectionResult,
          tags=["Image Analysis"],
          summary="Analyze an image for deepfake content",
          description="""
          Analyze an uploaded image to detect potential deepfake content.
          Supports JPEG, PNG, and WebP formats up to 10MB.
          """)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    model_preference: Optional[str] = Query(
        None, 
        description="Preferred model to use for analysis. If not specified, the best available model will be used."
    ),
    detailed: bool = Query(
        False, 
        description="Whether to include detailed analysis in the response"
    )
):
    """
    Analyze an image for potential deepfake content.
    
    This endpoint processes the uploaded image through multiple deepfake detection models
    and returns a confidence score indicating the likelihood of the image being AI-generated.
    """
    # Implementation would go here
    pass

@app.post("/api/v1/analyze/video", 
          response_model=DetectionResult,
          tags=["Video Analysis"],
          summary="Analyze a video for deepfake content",
          description="""
          Analyze an uploaded video to detect potential deepfake content.
          Supports MP4, AVI, and MOV formats up to 100MB.
          """)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    model_preference: Optional[str] = Query(None),
    detailed: bool = Query(False)
):
    # Implementation would go here
    pass

# Add more endpoint documentation as needed

# This would be used in your main FastAPI app like this:
# from fastapi import FastAPI
# from .docs.openapi_docs import app as docs_app
# 
# app = FastAPI()
# app.include_router(docs_app.router)

# The actual implementation would be in your route files (image.py, video.py, etc.)
# and would be imported and included in your main FastAPI application.
