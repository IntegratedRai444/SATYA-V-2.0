"""
Mock Services for Testing

Provides mock implementations of external services and APIs.
"""
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io

class MockModel:
    """Mock model for testing"""
    
    def __init__(self, model_name: str = "efficientnet_b7"):
        self.model_name = model_name
        self.version = "1.0.0"
        self.is_loaded = True
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Mock prediction"""
        # For images
        if isinstance(input_data, (np.ndarray, Image.Image)):
            return {
                "is_fake": np.random.random() > 0.5,
                "confidence": float(np.random.random()),
                "details": {
                    "model": self.model_name,
                    "version": self.version,
                    "input_shape": getattr(input_data, "shape", None) or getattr(input_data, "size", None)
                }
            }
        # For file paths
        elif isinstance(input_data, (str, Path)):
            return {
                "is_fake": np.random.random() > 0.5,
                "confidence": float(np.random.random()),
                "details": {
                    "model": self.model_name,
                    "version": self.version,
                    "file_path": str(input_data)
                }
            }
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

class MockStorageService:
    """Mock storage service for testing"""
    
    def __init__(self):
        self.files: Dict[str, bytes] = {}
    
    def upload_file(self, file_data: bytes, file_path: str, content_type: str = None) -> str:
        """Mock file upload"""
        file_id = str(hash(file_data))
        self.files[file_id] = file_data
        return file_id
    
    def download_file(self, file_id: str) -> bytes:
        """Mock file download"""
        return self.files.get(file_id)
    
    def delete_file(self, file_id: str) -> bool:
        """Mock file deletion"""
        if file_id in self.files:
            del self.files[file_id]
            return True
        return False

class MockAuthService:
    """Mock authentication service for testing"""
    
    def __init__(self):
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.next_user_id = 1
    
    def register_user(self, username: str, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """Mock user registration"""
        user_id = str(self.next_user_id)
        self.next_user_id += 1
        
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "is_active": True,
            "is_verified": False,
            "created_at": "2023-01-01T00:00:00",
            **kwargs
        }
        
        self.users[user_id] = {
            **user_data,
            "password_hash": f"hashed_{password}"
        }
        
        return user_data
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Mock user authentication"""
        for user in self.users.values():
            if (user["username"] == username or user["email"] == username) and \
               user["password_hash"] == f"hashed_{password}":
                return user
        return None
    
    def create_access_token(self, user_id: str) -> str:
        """Mock token creation"""
        token = f"mock_token_{len(self.tokens) + 1}"
        self.tokens[token] = {
            "user_id": user_id,
            "token": token,
            "expires_in": 3600
        }
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Mock token verification"""
        if token in self.tokens:
            user_id = self.tokens[token]["user_id"]
            return self.users.get(user_id)
        return None

class MockEmailService:
    """Mock email service for testing"""
    
    def __init__(self):
        self.sent_emails: List[Dict[str, Any]] = []
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        from_email: str = None,
        **kwargs
    ) -> bool:
        """Mock sending an email"""
        self.sent_emails.append({
            "to": to_email,
            "subject": subject,
            "html_content": html_content,
            "from_email": from_email,
            **kwargs
        })
        return True

def mock_file_upload(filename: str = "test.jpg", content_type: str = "image/jpeg") -> tuple:
    """Create a mock file upload for testing"""
    # Create a simple image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return (filename, img_byte_arr, content_type)

def apply_common_mocks():
    """Apply common mocks for testing"""
    return {
        "model_patch": patch('server.main.ModelManager', return_value=MockModel()),
        "storage_patch": patch('server.main.StorageService', return_value=MockStorageService()),
        "auth_patch": patch('server.main.AuthService', return_value=MockAuthService()),
        "email_patch": patch('server.main.EmailService', return_value=MockEmailService()),
    }
