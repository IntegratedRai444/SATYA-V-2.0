"""
Team Management Routes
API endpoints for team member management
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

try:
    from ..services.database import get_db_manager
    from .auth import UserResponse, get_current_user
except:
    from routes.auth import UserResponse, get_current_user
    from services.database import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class TeamMemberResponse(BaseModel):
    """Team member response model"""

    id: str
    email: str
    name: str
    role: str
    status: str
    joinDate: Optional[str] = None
    lastActive: Optional[str] = None


class InviteRequest(BaseModel):
    """Request model for inviting a team member"""

    email: EmailStr
    role: str  # admin, editor, viewer


class UpdateRoleRequest(BaseModel):
    """Request model for updating a team member's role"""

    role: str


@router.get("/members", response_model=List[TeamMemberResponse])
async def get_team_members():
    """
    Get all team members

    Returns:
        List of team members
    """
    try:
        db = get_db_manager()
        members = db.get_team_members()
        return members

    except Exception as e:
        logger.error(f"Error getting team members: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/invite")
async def invite_team_member(
    request: InviteRequest, current_user: UserResponse = Depends(get_current_user)
):
    """
    Invite a new team member

    Args:
        request: Invitation request with email and role
        current_user: Authenticated user sending the invite

    Returns:
        Success status and message
    """
    try:
        # Validate role
        valid_roles = ["admin", "editor", "viewer"]
        if request.role not in valid_roles:
            raise HTTPException(
                status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}"
            )

        invited_by_user_id = current_user.id

        db = get_db_manager()
        success = db.invite_team_member(
            email=request.email,
            role=request.role,
            invited_by_user_id=invited_by_user_id,
        )

        if success:
            return {"success": True, "message": f"Invitation sent to {request.email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create invitation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inviting team member: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/members/{user_id}/role")
async def update_team_member_role(user_id: int, request: UpdateRoleRequest):
    """
    Update a team member's role

    Args:
        user_id: ID of the user
        request: Update request with new role

    Returns:
        Success status and message
    """
    try:
        # Validate role
        valid_roles = ["admin", "editor", "viewer"]
        if request.role not in valid_roles:
            raise HTTPException(
                status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}"
            )

        db = get_db_manager()
        success = db.update_team_member_role(user_id, request.role)

        if success:
            return {"success": True, "message": f"Role updated to {request.role}"}
        else:
            raise HTTPException(status_code=404, detail="Team member not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating team member role: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/members/{user_id}")
async def remove_team_member(user_id: int):
    """
    Remove a team member

    Args:
        user_id: ID of the user to remove

    Returns:
        Success status and message
    """
    try:
        db = get_db_manager()
        success = db.remove_team_member(user_id)

        if success:
            return {"success": True, "message": "Team member removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Team member not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing team member: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
