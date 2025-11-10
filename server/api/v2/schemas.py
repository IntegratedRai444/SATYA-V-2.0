""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelName(str, Enum):
    EFFICIENTNET_B7 = "efficientnet_b7"
    XCEPTION = "xception"
    VIDEO_AUTH = "video_authenticator"
    AUDIO_DETECTOR = "audio_detector"

class BatchProcessRequest(BaseModel):
    """Request model for batch processing"""
    file_paths: List[str] = Field(
        ...,
        description="List of file paths to process",
        example=["path/to/file1.jpg", "path/to/file2.jpg"]
    )
    model_name: ModelName = Field(
        default=ModelName.EFFICIENTNET_B7,
        description="Model to use for processing"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Processing priority (1-10)"
    )

class BatchProcessResponse(BaseModel):
    """Response model for batch processing"""
    job_id: str = Field(..., description="Unique identifier for the batch job")
    status: str = Field(..., description="Current status of the job")

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    inference_time_ms: float = Field(..., gt=0)
    last_updated: datetime

class SystemHealth(BaseModel):
    """System health status"""
    status: str
    models_loaded: bool
    system: Dict[str, Any]
    timestamp: datetime

class AnalysisResult(BaseModel):
    """Analysis result for a single file"""
    file_path: str
    model_used: str
    is_fake: bool
    confidence: float = Field(..., ge=0, le=1)
    processing_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = {}

class ComparisonRequest(BaseModel):
    """Request model for media comparison"""
    file_paths: List[str] = Field(
        ...,
        min_items=2,
        description="At least two file paths to compare"
    )
    algorithm: str = Field(
        default="feature_matching",
        description="Comparison algorithm to use"
    )

class ComparisonResult(BaseModel):
    """Result of media comparison"""
    comparison_id: str
    similarity_scores: List[Dict[str, Any]]
    comparison_metrics: Dict[str, Any]

# User-related schemas
class UserBase(BaseModel):
    """Base user model"""
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    """User creation model"""
    password: str

class UserInDB(UserBase):
    """User model for database operations"""
    id: int
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Billing related schemas
class SubscriptionPlan(BaseModel):
    """Subscription plan details"""
    id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    max_requests: Optional[int] = None
    max_file_size: Optional[int] = None  # in bytes

class BillingInfo(BaseModel):
    """Billing information"""
    plan_id: str
    card_last4: Optional[str]
    card_brand: Optional[str]
    next_billing_date: Optional[datetime]
    status: str  # active, canceled, past_due, etc.

class Invoice(BaseModel):
    """Billing invoice"""
    id: str
    amount_due: float
    amount_paid: float
    currency: str
    created: datetime
    status: str
    pdf_url: Optional[HttpUrl] = None
