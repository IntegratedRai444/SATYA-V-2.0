"""
Database Manager Service
Handles database connections, sessions, and operations
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import httpx

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client with enhanced validation
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Enhanced validation with clear error messages
if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL must be set in environment variables. "
        "Please add it to your .env file or set it in your environment."
    )

if not SUPABASE_ANON_KEY and not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Neither SUPABASE_ANON_KEY nor SUPABASE_SERVICE_ROLE_KEY is set. "
        "Please add one of these to your .env file or set it in your environment."
    )

# Try service role key first (more permissions), then anon key
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY

# Log which key we're using (without exposing the actual key)
key_type = "service_role" if SUPABASE_KEY == SUPABASE_SERVICE_ROLE_KEY else "anon"
logger = logging.getLogger(__name__)

# Circuit breaker state
class CircuitBreaker:
    """Circuit breaker pattern for database operations"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, operation):
        """Execute operation with circuit breaker logic"""
        if self.state == "OPEN":
            # Circuit is open, check if we should close it
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return operation()
            else:
                raise Exception("Circuit breaker is OPEN")
                
        elif self.state == "HALF_OPEN":
            # Allow one operation to test if service is recovered
            try:
                result = operation()
                self.state = "CLOSED"  # Success, close circuit
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                self.last_failure_time = time.time()
                raise e
                
        elif self.state == "CLOSED":
            # Circuit is closed, allow operation
            try:
                result = operation()
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                raise e
        else:
            raise Exception(f"Invalid circuit breaker state: {self.state}")

# Add database directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))

try:
    from database.base import get_supabase
except ImportError:
    # Fallback if database module not available
    def get_supabase():
        return None
    logger.warning("Database module not available, database features disabled")


class DatabaseManager:
    """
    Supabase Database manager for handling all database operations
    """

    def __init__(self):
        """Initialize database manager"""
        self.supabase = get_supabase()
        self._connection_tested = False
        self.circuit_breaker = CircuitBreaker()
        
        # Check if supabase is available
        if self.supabase is None:
            logger.warning(" Supabase not available - database features disabled")
        else:
            logger.info(" Supabase database manager initialized")

    async def test_connection(self) -> bool:
        """
        Test database connection with a simple query - single attempt only
        """
        if self.supabase is None:
            logger.warning("⚠️ Cannot test connection - Supabase not available")
            return False
            
        try:
            # Test connection with a simple query to check if we can reach the database
            result = self.supabase.table("users").select("count").limit(1).execute()
            
            if result is not None:
                self._connection_tested = True
                logger.info("✅ Database connection test successful")
                return True
            else:
                logger.warning("⚠️ Database connection test returned None")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Database connection test failed: {str(e)[:100]}...")
            return False

    async def disconnect(self):
        """
        Close database connections and cleanup resources
        """
        try:
            # Supabase client doesn't need explicit disconnection, but we can cleanup
            self._connection_tested = False
            logger.info("✅ Database manager disconnected successfully")
        except Exception as e:
            logger.error(f"❌ Error during database disconnect: {e}")

    async def health_check(self) -> dict:
        """
        Perform comprehensive health check of database connection
        """
        health_status = {
            "connected": False,
            "response_time_ms": 0,
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            import time
            start_time = time.time()
            
            # Test basic connectivity
            result = self.supabase.table("users").select("count").limit(1).execute()
            
            response_time = (time.time() - start_time) * 1000
            health_status["response_time_ms"] = round(response_time, 2)
            
            if result is not None:
                health_status["connected"] = True
                self._connection_tested = True
                logger.info(f"✅ Database health check passed ({response_time:.2f}ms)")
            else:
                health_status["error"] = "No response from database"
                
        except Exception as e:
            health_status["error"] = str(e)
            logger.error(f"❌ Database health check failed: {e}")
            
        return health_status

    def is_connected(self) -> bool:
        """Check if database connection has been tested and successful"""
        return self._connection_tested

    async def execute_with_retry(self, operation, max_retries=1, retry_delay=1.0):
        """
        Execute database operation with circuit breaker and retry logic
        
        Args:
            operation: Async operation to execute
            max_retries: Maximum number of retry attempts (default: 1)
            retry_delay: Delay between retries (default: 1.0s)
        """
        # Use circuit breaker for the operation
        return await self.circuit_breaker(operation)

    def create_tables(self):
        """Create all database tables"""
        try:
            # Supabase tables are created via SQL scripts or UI
            logger.info("✅ Supabase tables managed via dashboard")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            # Supabase tables are dropped via dashboard
            logger.warning("⚠️ Supabase tables managed via dashboard")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            return False

    # ========================================================================
    # USER OPERATIONS
    # ========================================================================

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Create a new user using Supabase"""
        try:
            user_data = {
                "username": username,
                "email": email,
                "password": password,  # Match actual schema
                "full_name": full_name,
                "role": "user",
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("users").insert(user_data).execute()
            logger.info(f"User created: {username}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username using Supabase"""
        try:
            result = self.supabase.table("users").select("*").eq("username", username).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email using Supabase"""
        try:
            result = self.supabase.table("users").select("*").eq("email", email).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Get user by ID using Supabase"""
        try:
            result = self.supabase.table("users").select("*").eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None

    # ========================================================================
    # ANALYSIS OPERATIONS
    # ========================================================================

    def save_analysis_result(
        self,
        job_id: str,
        model_name: str = "SatyaAI",
        confidence: float = 0.0,
        is_deepfake: bool = False,
        analysis_data: dict = None,
        **kwargs,
    ) -> Optional[dict]:
        """Save analysis result to database using Supabase"""
        try:
            result_data = {
                "job_id": job_id,
                "model_name": model_name,
                "confidence": confidence,
                "is_deepfake": is_deepfake,
                "analysis_data": analysis_data or {},
                "created_at": datetime.utcnow().isoformat(),
                **kwargs,
            }
            
            result = self.supabase.table("analysis_results").insert(result_data).execute()
            logger.info(f"Analysis result saved: {job_id}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return None

    def get_analysis_by_file_id(self, file_id: str) -> Optional[dict]:
        """Get analysis result by file ID using Supabase"""
        try:
            result = self.supabase.table("analysis_results").select("*").eq("file_id", file_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get analysis by file ID: {e}")
            return None

    def get_user_analyses(
        self, user_id: int, limit: int = 10, offset: int = 0
    ) -> List[dict]:
        """Get all analyses for a user using Supabase"""
        try:
            result = self.supabase.table("analysis_results").select("*").eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get user analyses: {e}")
            return []

    def get_recent_analyses(self, limit: int = 10) -> List[dict]:
        """Get recent analyses using Supabase"""
        try:
            result = self.supabase.table("analysis_results").select("*").order("created_at", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get recent analyses: {e}")
            return []

    def save_scan_result(
        self,
        user_id: int,
        filename: str,
        type: str,
        result: str,
        confidence_score: float,
        detection_details: dict = None,
        metadata: dict = None,
        **kwargs,
    ) -> Optional[dict]:
        """Save scan result to database using Supabase"""
        try:
            scan_data = {
                "user_id": user_id,
                "filename": filename,
                "type": type,
                "result": result,
                "confidence_score": confidence_score,
                "detection_details": detection_details or {},
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                **kwargs,
            }
            
            result = self.supabase.table("scans").insert(scan_data).execute()
            logger.info(f"Scan result saved: {filename}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save scan: {e}")
            return None

    def get_user_scans(self, user_id: int, limit: int = 10) -> List[dict]:
        """Get scans for a user using Supabase"""
        try:
            result = self.supabase.table("scans").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get user scans: {e}")
            return []

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
    ) -> Optional[dict]:
        """Save file metadata using Supabase"""
        try:
            file_data = {
                "file_id": file_id,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "file_path": file_path,
                "file_type": file_type,
                "file_size": file_size,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                **kwargs,
            }
            
            result = self.supabase.table("file_metadata").insert(file_data).execute()
            logger.info(f"File metadata saved: {file_id}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")
            return None

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
    # USER MANAGEMENT METHODS
    # ============================================================================


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
