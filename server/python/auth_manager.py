"""
Authentication Manager for Python Server
Handles user authentication and session management
"""

import os
import jwt
import bcrypt
import sqlite3
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Authentication manager for the Python server.
    """
    
    def __init__(self, db_path: str = None, jwt_secret: str = None):
        """
        Initialize the auth manager.
        
        Args:
            db_path: Path to SQLite database
            jwt_secret: JWT secret key
        """
        self.db_path = db_path or os.environ.get('DATABASE_URL', './db.sqlite').replace('sqlite:', '')
        self.jwt_secret = jwt_secret or os.environ.get('JWT_SECRET', 'your-secret-key')
        self.active_sessions = {}
        
        logger.info(f"Auth manager initialized with database: {self.db_path}")
    
    def _get_db_connection(self):
        """Get database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, db=None) -> Dict[str, Any]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Password
            db: Optional database connection (for compatibility)
            
        Returns:
            Authentication result
        """
        try:
            conn = self._get_db_connection()
            if not conn:
                return {"success": False, "message": "Database connection failed"}
            
            # Query user
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, password, email, role FROM users WHERE username = ?",
                (username,)
            )
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                return {"success": False, "message": "Invalid credentials"}
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                return {"success": False, "message": "Invalid credentials"}
            
            # Generate session token
            session_token = self._generate_session_token(user)
            
            # Store session
            self.active_sessions[session_token] = {
                'user_id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=24)
            }
            
            return {
                "success": True,
                "message": "Authentication successful",
                "token": session_token,
                "user": {
                    "id": user['id'],
                    "username": user['username'],
                    "email": user['email'],
                    "role": user['role']
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "message": "Authentication failed"}
    
    def validate_session(self, token: str) -> Dict[str, Any]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Validation result
        """
        try:
            # Check if token exists in active sessions
            if token not in self.active_sessions:
                # Try to validate as JWT token
                return self._validate_jwt_token(token)
            
            session = self.active_sessions[token]
            
            # Check if session is expired
            if datetime.now() > session['expires_at']:
                del self.active_sessions[token]
                return {"valid": False, "message": "Session expired"}
            
            return {
                "valid": True,
                "user_id": session['user_id'],
                "username": session['username'],
                "email": session['email'],
                "role": session['role'],
                "is_admin": session['role'] == 'admin'
            }
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return {"valid": False, "message": "Session validation failed"}
    
    def _validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token (for compatibility with Node.js server).
        
        Args:
            token: JWT token
            
        Returns:
            Validation result
        """
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if user exists in database
            conn = self._get_db_connection()
            if not conn:
                return {"valid": False, "message": "Database connection failed"}
            
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, role FROM users WHERE id = ?",
                (payload['userId'],)
            )
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                return {"valid": False, "message": "User not found"}
            
            return {
                "valid": True,
                "user_id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "role": user['role'],
                "is_admin": user['role'] == 'admin'
            }
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "message": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "message": "Invalid token"}
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return {"valid": False, "message": "Token validation failed"}
    
    def logout(self, token: str) -> Dict[str, Any]:
        """
        Logout user by invalidating session.
        
        Args:
            token: Session token
            
        Returns:
            Logout result
        """
        try:
            if token in self.active_sessions:
                del self.active_sessions[token]
                return {"success": True, "message": "Logout successful"}
            else:
                return {"success": False, "message": "Session not found"}
                
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return {"success": False, "message": "Logout failed"}
    
    def get_active_sessions_count(self) -> int:
        """
        Get count of active sessions.
        
        Returns:
            Number of active sessions
        """
        # Clean expired sessions first
        self.clean_expired_sessions()
        return len(self.active_sessions)
    
    def clean_expired_sessions(self) -> int:
        """
        Clean expired sessions.
        
        Returns:
            Number of sessions cleaned
        """
        try:
            current_time = datetime.now()
            expired_tokens = []
            
            for token, session in self.active_sessions.items():
                if current_time > session['expires_at']:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.active_sessions[token]
            
            if expired_tokens:
                logger.info(f"Cleaned {len(expired_tokens)} expired sessions")
            
            return len(expired_tokens)
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            return 0
    
    def _generate_session_token(self, user) -> str:
        """
        Generate a session token for the user.
        
        Args:
            user: User record
            
        Returns:
            Session token
        """
        import secrets
        import hashlib
        
        # Create a unique token
        token_data = f"{user['id']}-{user['username']}-{datetime.now().isoformat()}-{secrets.token_hex(16)}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        
        return token


# Global auth manager instance
_auth_manager = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager instance."""
    global _auth_manager
    
    if _auth_manager is None:
        _auth_manager = AuthManager()
    
    return _auth_manager


# Convenience functions for backward compatibility
def authenticate_user(username: str, password: str, db=None) -> Dict[str, Any]:
    """Authenticate user - convenience function."""
    return get_auth_manager().authenticate_user(username, password, db)


def validate_session(token: str) -> Dict[str, Any]:
    """Validate session - convenience function."""
    return get_auth_manager().validate_session(token)


def logout(token: str) -> Dict[str, Any]:
    """Logout user - convenience function."""
    return get_auth_manager().logout(token)


def get_active_sessions_count() -> int:
    """Get active sessions count - convenience function."""
    return get_auth_manager().get_active_sessions_count()


def clean_expired_sessions() -> int:
    """Clean expired sessions - convenience function."""
    return get_auth_manager().clean_expired_sessions()