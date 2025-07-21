import os
import time
import json
import hashlib
import secrets
from datetime import datetime, timedelta

class AuthManager:
    """
    Manages user authentication and session handling for SatyaAI
    """
    
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour in seconds
    
    def authenticate_user(self, username, password, db_connection=None):
        """
        Authenticate a user against the database
        
        Args:
            username: User's username
            password: User's password
            db_connection: Database connection (optional)
            
        Returns:
            dict: Authentication result with session token if successful
        """
        # In a real implementation, this would verify against the database
        # For this demo, we'll simulate a database check
        
        if not username or not password:
            return {
                "success": False,
                "message": "Username and password are required"
            }
        
        # For a real implementation, we would:
        # 1. Query the database for the user
        # 2. Verify the password hash
        # 3. Create and store a session token
        
        # For demo purposes, accept any non-empty credentials
        session_token = self._generate_session_token()
        user_id = hashlib.md5(username.encode()).hexdigest()[:8]
        
        # Create session
        self.active_sessions[session_token] = {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=self.session_timeout),
            "is_admin": username.lower() == "admin"
        }
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "session_token": session_token,
            "expires_at": self.active_sessions[session_token]["expires_at"].isoformat()
        }
    
    def validate_session(self, session_token):
        """
        Validate a session token
        
        Args:
            session_token: Session token to validate
            
        Returns:
            dict: Session validation result
        """
        if not session_token or session_token not in self.active_sessions:
            return {
                "valid": False,
                "message": "Invalid session token"
            }
        
        session = self.active_sessions[session_token]
        
        # Check if session has expired
        if datetime.now() > session["expires_at"]:
            # Clean up expired session
            del self.active_sessions[session_token]
            
            return {
                "valid": False,
                "message": "Session expired"
            }
        
        # Extend session expiration time
        session["expires_at"] = datetime.now() + timedelta(seconds=self.session_timeout)
        
        return {
            "valid": True,
            "user_id": session["user_id"],
            "username": session["username"],
            "is_admin": session["is_admin"],
            "expires_at": session["expires_at"].isoformat()
        }
    
    def logout(self, session_token):
        """
        Invalidate a session token
        
        Args:
            session_token: Session token to invalidate
            
        Returns:
            dict: Logout result
        """
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
            return {
                "success": True,
                "message": "Successfully logged out"
            }
        
        return {
            "success": False,
            "message": "Invalid session token"
        }
    
    def _generate_session_token(self):
        """
        Generate a secure random session token
        
        Returns:
            str: Session token
        """
        return secrets.token_hex(32)
    
    def clean_expired_sessions(self):
        """
        Clean up expired sessions
        
        Returns:
            int: Number of sessions cleaned up
        """
        now = datetime.now()
        expired_tokens = [
            token for token, session in self.active_sessions.items()
            if now > session["expires_at"]
        ]
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        return len(expired_tokens)
    
    def get_active_sessions_count(self):
        """
        Get the number of active sessions
        
        Returns:
            int: Number of active sessions
        """
        return len(self.active_sessions)

# Create singleton instance
auth_manager = AuthManager()

# Export functions for the API
def authenticate_user(username, password, db_connection=None):
    return auth_manager.authenticate_user(username, password, db_connection)

def validate_session(session_token):
    return auth_manager.validate_session(session_token)

def logout(session_token):
    return auth_manager.logout(session_token)

def get_active_sessions_count():
    return auth_manager.get_active_sessions_count()

def clean_expired_sessions():
    return auth_manager.clean_expired_sessions()