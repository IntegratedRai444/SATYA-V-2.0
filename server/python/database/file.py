"""
File Metadata Model
SQLAlchemy model for file management
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from database.base import Base


class FileMetadata(Base):
    """File metadata model for tracking uploaded files"""
    
    __tablename__ = "file_metadata"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # File information
    file_id = Column(String(100), unique=True, index=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # File details
    file_type = Column(String(20), nullable=False)  # image, video, audio
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=False)  # in bytes
    
    # File hash for deduplication
    file_hash = Column(String(64), nullable=True, index=True)  # SHA-256 hash
    
    # Status
    is_analyzed = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    analyzed_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<FileMetadata(id={self.id}, filename='{self.original_filename}', type='{self.file_type}')>"
