"""
User Feedback Model
SQLAlchemy model for storing user feedback on analysis results
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

try:
    from database.base import Base
except:
    from .base import Base


class FeedbackType(enum.Enum):
    """Enum for feedback types"""
    CORRECT = "correct"  # User confirms the prediction was correct
    INCORRECT = "incorrect"  # User says the prediction was wrong
    UNCERTAIN = "uncertain"  # User is not sure


class UserFeedback(Base):
    """User feedback model for tracking ground truth"""
    
    __tablename__ = "user_feedback"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to analysis result
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False, index=True)
    
    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Feedback data
    feedback_type = Column(String(20), nullable=False)  # correct, incorrect, uncertain
    
    # Ground truth (what the user says is the actual truth)
    actual_label = Column(String(50), nullable=True)  # authentic, deepfake, suspicious
    
    # Optional comment from user
    comment = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    # analysis = relationship("AnalysisResult", back_populates="feedback")
    
    def __repr__(self):
        return f"<UserFeedback(id={self.id}, analysis_id={self.analysis_id}, feedback_type='{self.feedback_type}')>"
