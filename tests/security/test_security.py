"""
Security Testing Suite

This module contains tests for security vulnerabilities including:
- SQL Injection
- XSS (Cross-Site Scripting)
- CSRF (Cross-Site Request Forgery)
- Authentication Bypass
- Rate Limiting
- Security Headers
- File Upload Vulnerabilities
- API Security
"""
import pytest
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

# Test data for security tests
MALICIOUS_PAYLOADS = {
    'sql_injection': [
        "' OR '1'='1",
        '" OR "1"="1',
        "1; DROP TABLE users; --",
        "1' UNION SELECT username, password FROM users --",
        "1' OR 1=1 --",
        "1' OR 'a'='a",
        "1' OR 'a'='a' --",
        "1' OR 'a'='a' /*",
        "1' OR 1=1 #",
        "1' OR '1'='1' /*",
    ],
    'xss': [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<a href='javascript:alert(1)'>click</a>",
        "<iframe src='javascript:alert(1)'>",
        "<object data='javascript:alert(1)'>",
        "<img src='x' onerror='alert(1)'>",
        "<div onmouseover='alert(1)'>test</div>",
        "<img src='x' onerror=alert('XSS')>"
    ],
    'path_traversal': [
        "../../../etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..\\..\\..\\windows\\win.ini",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%5c..%5c..%5cboot.ini"
    ]
}

class TestAPISecurity:
    """Tests for API security vulnerabilities"""
    
    def test_sql_injection(self, client, auth_headers):
        """Test for SQL injection vulnerabilities in API endpoints"""
        # Test login endpoint
        for payload in MALICIOUS_PAYLOADS['sql_injection']:
            response = client.post(
                "/api/v1/auth/login",
                data={"username": payload, "password": "test"}
            )
            # Should not return 500 error for SQL errors
            assert response.status_code != 500, f"SQLi vulnerability found with payload: {payload}"
            
            # Test other endpoints that accept user input
            endpoints = [
                ("/api/v1/analysis/search", {"query": payload}),
                ("/api/v1/users/search", {"q": payload}),
            ]
            
            for endpoint, params in endpoints:
                response = client.get(
                    endpoint,
                    params=params,
                    headers=auth_headers
                )
                assert response.status_code != 500, f"SQLi in {endpoint} with payload: {payload}"
    
    def test_xss_protection(self, client, auth_headers):
        """Test for XSS vulnerabilities in API responses"""
        for payload in MALICIOUS_PAYLOADS['xss']:
            # Test search endpoint
            response = client.get(
                "/api/v1/analysis/search",
                params={"query": payload},
                headers=auth_headers
            )
            
            # Check if response contains the payload (should be escaped)
            response_text = response.text
            if payload in response_text and '<script>' in payload.lower():
                # If we find unescaped script tags, it's a potential XSS
                assert False, f"XSS vulnerability found with payload: {payload}"
    
    def test_rate_limiting(self, client, auth_headers):
        """Test API rate limiting"""
        endpoint = "/api/v1/analysis/history"
        
        # Make requests until rate limited
        rate_limited = False
        for _ in range(100):  # Should hit rate limit before this
            response = client.get(endpoint, headers=auth_headers)
            if response.status_code == 429:
                rate_limited = True
                break
        
        assert rate_limited, "Rate limiting not working"
    
    def test_security_headers(self, client):
        """Test for security-related HTTP headers"""
        response = client.get("/")
        
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": None,  # Just check if it exists
            "Strict-Transport-Security": None,  # Just check if it exists
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": None  # Just check if it exists
        }
        
        for header, expected_value in required_headers.items():
            assert header in response.headers, f"Missing security header: {header}"
            if expected_value:
                assert response.headers[header] == expected_value, f"Invalid value for {header}"

class TestFileUploadSecurity:
    """Tests for file upload security"""
    
    def test_malicious_file_upload(self, client, auth_headers, tmp_path):
        """Test uploading potentially malicious files"""
        # Create a test file with malicious content
        malicious_files = [
            ("test.php", "<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("test.html", "<script>alert('XSS')</script>", "text/html"),
            ("test.exe", b"MZ\x90\x00\x03\x00\x00\x00\x04", "application/x-msdownload"),
        ]
        
        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            response = client.post(
                "/api/v1/analyze/image",
                files=files,
                headers=auth_headers
            )
            
            # Should reject malicious files with 400 Bad Request
            assert response.status_code in [400, 415], f"Accepted malicious file: {filename}"
    
    def test_large_file_upload(self, client, auth_headers, tmp_path):
        """Test uploading very large files"""
        # Create a large file (11MB)
        large_file = tmp_path / "large.jpg"
        with open(large_file, "wb") as f:
            f.seek(10 * 1024 * 1024)  # 10MB
            f.write(b"0")
        
        with open(large_file, "rb") as f:
            response = client.post(
                "/api/v1/analyze/image",
                files={"file": ("large.jpg", f, "image/jpeg")},
                headers=auth_headers
            )
            
            # Should reject files larger than 10MB
            assert response.status_code == 413, "Accepted file larger than size limit"

class TestAuthenticationSecurity:
    """Tests for authentication security"""
    
    def test_jwt_security(self, client):
        """Test JWT token security"""
        # Register a test user
        user_data = {
            "email": "security_test@example.com",
            "password": "SecurePass123!",
            "username": "security_test"
        }
        client.post("/api/v1/auth/register", json=user_data)
        
        # Login to get token
        response = client.post(
            "/api/v1/auth/login",
            data={"username": user_data["email"], "password": user_data["password"]}
        )
        token = response.json()["access_token"]
        
        # Test token validation
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        
        # Test with modified token
        modified_token = token[:-5] + "abcde"
        headers = {"Authorization": f"Bearer {modified_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 401, "Accepted modified JWT token"
        
        # Test with expired token (if we can simulate it)
        # This would typically require mocking time in tests
    
    def test_password_policy(self, client):
        """Test password complexity requirements"""
        weak_passwords = [
            "password",  # Common password
            "123456",    # Too short
            "qwerty",    # Common password
            "abcdefg",   # Sequential letters
            "111111",    # Repeated characters
        ]
        
        for password in weak_passwords:
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"test_{password}@example.com",
                    "password": password,
                    "username": f"test_{password}"
                }
            )
            assert response.status_code == 422, f"Accepted weak password: {password}"

class TestAPIAccessControl:
    """Tests for API access control"""
    
    def test_unauthenticated_access(self, client):
        """Test that protected endpoints require authentication"""
        protected_endpoints = [
            ("/api/v1/analysis/history", "GET"),
            ("/api/v1/analysis/123", "GET"),
            ("/api/v1/users/me", "GET"),
        ]
        
        for endpoint, method in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint)
            else:
                continue
                
            assert response.status_code in [401, 403], f"Unauthenticated access allowed to {endpoint}"
    
    def test_role_based_access_control(self, client, auth_headers):
        """Test that users can only access resources they own"""
        # Get current user info
        response = client.get("/api/v1/users/me", headers=auth_headers)
        current_user_id = response.json()["id"]
        
        # Try to access another user's data
        other_user_id = "00000000-0000-0000-0000-000000000000"
        
        # Should not be able to access another user's data
        response = client.get(
            f"/api/v1/users/{other_user_id}",
            headers=auth_headers
        )
        assert response.status_code == 403, "Accessed another user's data"
        
        # Should be able to access own data
        response = client.get(
            f"/api/v1/users/{current_user_id}",
            headers=auth_headers
        )
        assert response.status_code == 200, "Cannot access own data"
