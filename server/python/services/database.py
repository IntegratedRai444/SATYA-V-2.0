"""
Database Manager Service
Handles database connections, sessions, and operations
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

# Add database directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))

from database.analysis import AnalysisResult
from database.base import get_supabase
from database.feedback import UserFeedback
from database.file import FileMetadata
from database.team import TeamInvitation, TeamMember
from database.user import User

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for handling all database operations
    """

    def __init__(self):
        """Initialize database manager"""
        self.engine = engine
        self.SessionLocal = SessionLocal
        logger.info("Database manager initialized")

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("⚠️ All database tables dropped")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            return False

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    # ========================================================================
    # USER OPERATIONS
    # ========================================================================

    def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
    ) -> Optional[User]:
        """Create a new user"""
        db = self.get_session()
        try:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                created_at=datetime.utcnow(),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"User created: {username}")
            return user
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create user: {e}")
            return None
        finally:
            db.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        db = self.get_session()
        try:
            return db.query(User).filter(User.username == username).first()
        finally:
            db.close()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        db = self.get_session()
        try:
            return db.query(User).filter(User.email == email).first()
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        db = self.get_session()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    # ========================================================================
    # ANALYSIS OPERATIONS
    # ========================================================================

    def save_analysis_result(
        self,
        file_id: str,
        file_name: str,
        file_type: str,
        authenticity_score: float,
        label: str,
        confidence: float,
        details: dict,
        user_id: Optional[int] = None,
        **kwargs,
    ) -> Optional[AnalysisResult]:
        """Save analysis result to database"""
        db = self.get_session()
        try:
            analysis = AnalysisResult(
                user_id=user_id,
                file_id=file_id,
                file_name=file_name,
                file_type=file_type,
                authenticity_score=authenticity_score,
                label=label,
                confidence=confidence,
                details=details,
                created_at=datetime.utcnow(),
                **kwargs,
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            logger.info(f"Analysis result saved: {file_id}")
            return analysis
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save analysis: {e}")
            return None
        finally:
            db.close()

    def get_analysis_by_file_id(self, file_id: str) -> Optional[AnalysisResult]:
        """Get analysis result by file ID"""
        db = self.get_session()
        try:
            return (
                db.query(AnalysisResult)
                .filter(AnalysisResult.file_id == file_id)
                .first()
            )
        finally:
            db.close()

    def get_user_analyses(
        self, user_id: int, limit: int = 10, offset: int = 0
    ) -> List[AnalysisResult]:
        """Get all analyses for a user"""
        db = self.get_session()
        try:
            return (
                db.query(AnalysisResult)
                .filter(AnalysisResult.user_id == user_id)
                .order_by(AnalysisResult.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        finally:
            db.close()

    def get_recent_analyses(self, limit: int = 10) -> List[AnalysisResult]:
        """Get recent analyses"""
        db = self.get_session()
        try:
            return (
                db.query(AnalysisResult)
                .order_by(AnalysisResult.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    def get_analysis_stats(self) -> dict:
        """Get analysis statistics"""
        db = self.get_session()
        try:
            total = db.query(AnalysisResult).count()

            # Count by type
            image_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.file_type == "image")
                .count()
            )
            video_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.file_type == "video")
                .count()
            )
            audio_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.file_type == "audio")
                .count()
            )
            text_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.file_type == "text")
                .count()
            )

            # Count by label
            authentic_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.label == "authentic")
                .count()
            )
            deepfake_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.label == "deepfake")
                .count()
            )
            suspicious_count = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.label == "suspicious")
                .count()
            )

            # Average processing time
            avg_time = (
                db.query(func.avg(AnalysisResult.processing_time)).scalar() or 0.0
            )

            return {
                "total_analyses": total,
                "avg_processing_time": float(avg_time),
                "by_type": {
                    "image": image_count,
                    "video": video_count,
                    "audio": audio_count,
                    "text": text_count,
                },
                "by_result": {
                    "authentic": authentic_count,
                    "deepfake": deepfake_count,
                    "suspicious": suspicious_count,
                },
            }
        finally:
            db.close()

    def get_daily_analysis_stats(self, days: int = 7) -> dict:
        """Get daily analysis statistics for the last N days"""
        db = self.get_session()
        try:
            from datetime import datetime, timedelta

            from sqlalchemy import case, func

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Query for daily counts
            daily_stats = (
                db.query(
                    func.date(AnalysisResult.created_at).label("date"),
                    func.count(AnalysisResult.id).label("total"),
                    func.sum(
                        case((AnalysisResult.label == "deepfake", 1), else_=0)
                    ).label("deepfakes"),
                )
                .filter(AnalysisResult.created_at >= start_date)
                .group_by(func.date(AnalysisResult.created_at))
                .all()
            )

            # Format results
            results = {"dates": [], "scans": [], "deepfakes": []}

            # Fill in missing days with 0
            stats_map = {str(stat.date): stat for stat in daily_stats}

            for i in range(days):
                date = (end_date - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                results["dates"].append(date)

                if date in stats_map:
                    stat = stats_map[date]
                    results["scans"].append(stat.total)
                    results["deepfakes"].append(int(stat.deepfakes or 0))
                else:
                    results["scans"].append(0)
                    results["deepfakes"].append(0)

            return results
        finally:
            db.close()

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    def save_file_metadata(
        self,
        file_id: str,
        original_filename: str,
        stored_filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        user_id: Optional[int] = None,
        **kwargs,
    ) -> Optional[FileMetadata]:
        """Save file metadata"""
        db = self.get_session()
        try:
            file_meta = FileMetadata(
                user_id=user_id,
                file_id=file_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                created_at=datetime.utcnow(),
                **kwargs,
            )
            db.add(file_meta)
            db.commit()
            db.refresh(file_meta)
            logger.info(f"File metadata saved: {file_id}")
            return file_meta
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save file metadata: {e}")
            return None
        finally:
            db.close()

    def get_active_users_count(self, days: int = 30) -> int:
        """Get count of active users (who analyzed files) in last N days"""
        db = self.get_session()
        try:
            from datetime import datetime, timedelta

            from sqlalchemy import distinct, func

            start_date = datetime.now() - timedelta(days=days)

            # Count users who have analysis results in the period
            count = (
                db.query(func.count(distinct(AnalysisResult.user_id)))
                .filter(AnalysisResult.created_at >= start_date)
                .scalar()
            )

            return count or 0
        finally:
            db.close()

    def get_total_storage_used(self) -> str:
        """Get total storage used by uploaded files"""
        db = self.get_session()
        try:
            from sqlalchemy import func

            # Sum file_size from FileMetadata
            total_bytes = db.query(func.sum(FileMetadata.file_size)).scalar() or 0

            # Convert to readable format
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if total_bytes < 1024.0:
                    return f"{total_bytes:.1f} {unit}"
                total_bytes /= 1024.0

            return f"{total_bytes:.1f} PB"
        finally:
            db.close()

    def get_accuracy_rate(self) -> float:
        """Calculate system accuracy rate based on high confidence predictions"""
        db = self.get_session()
        try:
            from sqlalchemy import func

            # Get average confidence of all predictions
            # This is a proxy for accuracy in the absence of ground truth
            avg_conf = db.query(func.avg(AnalysisResult.confidence)).scalar() or 0.0

            # Convert to percentage (0-100)
            return float(avg_conf * 100)
        finally:
            db.close()

    def save_user_feedback(
        self,
        analysis_id: int,
        feedback_type: str,
        actual_label: str = None,
        comment: str = None,
        user_id: int = None,
    ) -> bool:
        """
        Save user feedback for an analysis result

        Args:
            analysis_id: ID of the analysis result
            feedback_type: Type of feedback (correct, incorrect, uncertain)
            actual_label: What the user says is the actual truth
            comment: Optional comment from user
            user_id: ID of the user providing feedback

        Returns:
            True if saved successfully
        """
        db = self.get_session()
        try:
            feedback = UserFeedback(
                analysis_id=analysis_id,
                user_id=user_id,
                feedback_type=feedback_type,
                actual_label=actual_label,
                comment=comment,
                created_at=datetime.utcnow(),
            )
            db.add(feedback)
            db.commit()
            logger.info(f"✅ User feedback saved for analysis {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user feedback: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def get_false_positive_rate(self) -> float:
        """
        Calculate false positive rate based on user feedback

        False Positive = Model predicted deepfake, but user says it's authentic
        FPR = False Positives / (False Positives + True Negatives)

        Returns:
            False positive rate (0.0 to 1.0), or 0.05 if insufficient data
        """
        db = self.get_session()
        try:
            from sqlalchemy import and_

            # Get all feedback with ground truth
            feedbacks = (
                db.query(UserFeedback, AnalysisResult)
                .join(AnalysisResult, UserFeedback.analysis_id == AnalysisResult.id)
                .filter(UserFeedback.feedback_type == "incorrect")
                .all()
            )

            if len(feedbacks) < 10:
                # Not enough data, return default
                logger.warning(
                    "⚠️ Insufficient feedback data for FPR calculation, using default 0.05"
                )
                return 0.05

            # Count false positives
            # FP = Model said deepfake, user said authentic
            false_positives = 0
            true_negatives = 0

            for feedback, analysis in feedbacks:
                model_predicted_fake = analysis.label.lower() in [
                    "deepfake",
                    "suspicious",
                ]
                user_says_authentic = (
                    feedback.actual_label
                    and feedback.actual_label.lower() == "authentic"
                )
                user_says_fake = (
                    feedback.actual_label
                    and feedback.actual_label.lower() in ["deepfake", "fake"]
                )

                if model_predicted_fake and user_says_authentic:
                    false_positives += 1
                elif not model_predicted_fake and user_says_authentic:
                    true_negatives += 1

            # Calculate FPR
            denominator = false_positives + true_negatives
            if denominator == 0:
                return 0.05

            fpr = false_positives / denominator
            logger.info(
                f"📊 Calculated FPR: {fpr:.4f} (FP={false_positives}, TN={true_negatives})"
            )
            return float(fpr)

        except Exception as e:
            logger.error(f"Failed to calculate FPR: {e}")
            return 0.05
        finally:
            db.close()

    # ============================================================================
    # TEAM MANAGEMENT METHODS
    # ============================================================================

    def get_team_members(self) -> List[Dict]:
        """Get all team members"""
        db = self.get_session()
        try:
            members = (
                db.query(TeamMember, User)
                .join(User, TeamMember.user_id == User.id)
                .filter(TeamMember.status == "active")
                .all()
            )

            result = []
            for team_member, user in members:
                result.append(
                    {
                        "id": str(user.id),
                        "email": user.email,
                        "name": user.username,
                        "role": team_member.role,
                        "status": team_member.status,
                        "joinDate": team_member.joined_at.isoformat()
                        if team_member.joined_at
                        else None,
                        "lastActive": user.last_login.isoformat()
                        if user.last_login
                        else None,
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Failed to get team members: {e}")
            return []
        finally:
            db.close()

    def invite_team_member(
        self, email: str, role: str, invited_by_user_id: int
    ) -> bool:
        """
        Invite a new team member

        Args:
            email: Email of the person to invite
            role: Role to assign (admin, editor, viewer)
            invited_by_user_id: ID of user sending invitation

        Returns:
            True if invitation created successfully
        """
        db = self.get_session()
        try:
            import secrets
            from datetime import datetime, timedelta

            # Generate invitation token
            token = secrets.token_urlsafe(32)

            # Set expiration (7 days from now)
            expires_at = datetime.utcnow() + timedelta(days=7)

            # Create invitation
            invitation = TeamInvitation(
                email=email,
                role=role,
                invited_by_user_id=invited_by_user_id,
                invitation_token=token,
                expires_at=expires_at,
                status="pending",
                created_at=datetime.utcnow(),
            )

            db.add(invitation)
            db.commit()

            logger.info(f"✅ Team invitation created for {email}")
            return True

        except Exception as e:
            logger.error(f"Failed to create invitation: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def update_team_member_role(self, user_id: int, new_role: str) -> bool:
        """
        Update a team member's role

        Args:
            user_id: ID of the user
            new_role: New role to assign

        Returns:
            True if updated successfully
        """
        db = self.get_session()
        try:
            team_member = (
                db.query(TeamMember).filter(TeamMember.user_id == user_id).first()
            )

            if not team_member:
                logger.warning(f"Team member not found for user_id {user_id}")
                return False

            team_member.role = new_role
            db.commit()

            logger.info(f"✅ Updated role for user {user_id} to {new_role}")
            return True

        except Exception as e:
            logger.error(f"Failed to update team member role: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def remove_team_member(self, user_id: int) -> bool:
        """
        Remove a team member (set status to suspended)

        Args:
            user_id: ID of the user to remove

        Returns:
            True if removed successfully
        """
        db = self.get_session()
        try:
            team_member = (
                db.query(TeamMember).filter(TeamMember.user_id == user_id).first()
            )

            if not team_member:
                logger.warning(f"Team member not found for user_id {user_id}")
                return False

            team_member.status = "suspended"
            db.commit()

            logger.info(f"✅ Removed team member {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove team member: {e}")
            db.rollback()
            return False
        finally:
            db.close()


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
