"""
Analysis Result Model
SQLAlchemy model for storing ML analysis results
"""

from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.base import Base


class AnalysisResult(Base):
    """Analysis result model for storing deepfake detection results"""
    
    __tablename__ = "analysis_results"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # File information
    file_id = Column(String(100), unique=True, index=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # image, video, audio, text
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # in bytes
    
    # Analysis results
    authenticity_score = Column(Float, nullable=False)
    label = Column(String(50), nullable=False)  # authentic, deepfake, suspicious
    confidence = Column(Float, nullable=False)
    
    # Detailed results (JSON)
    details = Column(JSON, nullable=True)
    
    # ML model info
    model_version = Column(String(50), nullable=True)
    detector_type = Column(String(50), nullable=True)  # image, video, audio, text, multimodal
    
    # Processing info
    processing_time = Column(Float, nullable=True)  # in seconds
    
    # Additional notes
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    # user = relationship("User", back_populates="analyses")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, file_type='{self.file_type}', label='{self.label}', score={self.authenticity_score})>"
