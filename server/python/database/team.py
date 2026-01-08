"""
Team Management Models
SQLAlchemy models for team and invitation management
"""

import enum

from sqlalchemy import (Boolean, Column, DateTime, Enum, ForeignKey, Integer,
                        String)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

try:
    from database.base import Base
except:
    from .base import Base


class TeamMemberStatus(enum.Enum):
    """Enum for team member status"""

    ACTIVE = "active"
    PENDING = "pending"
    SUSPENDED = "suspended"


class TeamMember(Base):
    """Team member model"""

    __tablename__ = "team_members"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Team information
    role = Column(String(20), nullable=False, default="viewer")  # admin, editor, viewer
    status = Column(
        String(20), nullable=False, default="active"
    )  # active, pending, suspended

    # Timestamps
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<TeamMember(id={self.id}, user_id={self.user_id}, role='{self.role}', status='{self.status}')>"


class TeamInvitation(Base):
    """Team invitation model"""

    __tablename__ = "team_invitations"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Invitation details
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(20), nullable=False, default="viewer")

    # Invited by
    invited_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Status
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, accepted, expired

    # Token for invitation link
    invitation_token = Column(String(255), unique=True, nullable=False, index=True)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    accepted_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<TeamInvitation(id={self.id}, email='{self.email}', role='{self.role}', status='{self.status}')>"
