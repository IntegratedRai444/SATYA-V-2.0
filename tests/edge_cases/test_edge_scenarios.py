"""
Edge Case Testing for Robustness
"""
import os
import pytest
import json
from pathlib import Path
from playwright.sync_api import Page, expect
from typing import Dict, Any, List, Optional
import random
import string

class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.test_files_dir = Path(__file__).parent / "test_files"
        self.test_files_dir.mkdir(exist_ok=True)
    
    def generate_random_string(self, length: int = 10) -> str:
        """Generate a random string of fixed length"""
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))
    
    def create_large_file(self, size_mb: int, extension: str = "txt") -> Path:
        """Create a large file of specified size in MB"""
        file_path = self.test_files_dir / f"large_file_{size_mb}mb.{extension}"
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(file_path, 'wb') as f:
            for _ in range(size_mb):
                f.write(os.urandom(chunk_size))
        
        return file_path
    
    def test_oversized_file_upload(self, page: Page):
        """Test handling of files larger than the maximum allowed size"""
        # Create a 21MB file (assuming 20MB limit)
        large_file = self.create_large_file(21, "bin")
        
        try:
            # Navigate to upload page
            page.goto(f"{self.base_url}/upload")
            
            # Upload the large file
            with page.expect_file_chooser() as fc_info:
                page.click("input[type=file]")
            file_chooser = fc_info.value
            file_chooser.set_files(str(large_file))
            
            # Verify error message
            error_message = page.locator(".error-message")
            expect(error_message).to_contain_text("too large")
            
            # Verify submit button is disabled
            submit_button = page.locator("button[type=submit]")
            expect(submit_button).to_be_disabled()
            
        finally:
            # Clean up
            if large_file.exists():
                large_file.unlink()
    
    def test_malicious_file_upload(self, page: Page):
        """Test handling of potentially malicious files"""
        # Create a file with a double extension
        malicious_file = self.test_files_dir / "malicious.exe.jpg"
        malicious_file.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04")
        
        try:
            # Navigate to upload page
            page.goto(f"{self.base_url}/upload")
            
            # Upload the file
            with page.expect_file_chooser() as fc_info:
                page.click("input[type=file]")
            file_chooser = fc_info.value
            file_chooser.set_files(str(malicious_file))
            
            # Verify the file was rejected
            error_message = page.locator(".error-message")
            expect(error_message).to_contain_text("not allowed")
            
        finally:
            if malicious_file.exists():
                malicious_file.unlink()
    
    def test_sql_injection(self, page: Page):
        """Test SQL injection attempts in form fields"""
        # Navigate to login page
        page.goto(f"{self.base_url}/login")
        
        # SQL injection attempts
        sql_payloads = [
            "' OR '1'='1",
            ""; DROP TABLE users; --",
            "admin' --",
            "1' OR '1' = '1",
            "1' EXEC XP_"
        ]
        
        for payload in sql_payloads:
            # Fill form with SQL injection
            page.fill("input[name=username]", payload)
            page.fill("input[name=password]", "password123")
            
            # Submit form
            with page.expect_response(lambda response: 
                response.status == 400 or  # Bad Request
                response.status == 401     # Unauthorized
            ) as response_info:
                page.click("button[type=submit]")
            
            # Verify the response indicates failure
            response = response_info.value
            assert response.status in (400, 401), \
                f"SQL injection attempt '{payload}' did not fail as expected"
    
    def test_xss_attempts(self, page: Page):
        """Test XSS (Cross-Site Scripting) attempts"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src='x' onerror='alert(1)'>",
            "<svg/onload=alert('XSS')>",
            "{x:alert(1)}"
        ]
        
        # Test in search field
        page.goto(f"{self.base_url}/search")
        
        for payload in xss_payloads:
            # Enter XSS payload in search field
            search_input = page.locator("input[name=search]")
            search_input.fill(payload)
            
            # Submit the form
            with page.expect_response(lambda response: 
                response.status == 400 or  # Bad Request
                response.status == 200      # OK (but sanitized)
            ) as response_info:
                page.click("button[type=submit]")
            
            # Check if the page contains the raw payload (it shouldn't)
            page_content = page.content()
            assert payload not in page_content, \
                f"XSS payload '{payload}' was not properly sanitized"
    
    def test_rate_limiting(self, page: Page):
        """Test API rate limiting"""
        # Make multiple rapid requests to a rate-limited endpoint
        endpoint = f"{self.base_url}/api/analyze"
        
        # First request should succeed
        response = page.goto(endpoint, wait_until="networkidle")
        assert response.status == 200 or response.status == 401  # 401 if auth required
        
        # Make several more requests quickly
        for _ in range(5):
            response = page.goto(endpoint, wait_until="networkidle")
        
        # Next request should be rate limited (429 Too Many Requests)
        response = page.goto(endpoint, wait_until="networkidle")
        assert response.status == 429, "Rate limiting not working as expected"
    
    def test_invalid_content_type(self, page: Page):
        """Test handling of invalid content types"""
        # Try to upload a file with an invalid content type
        test_file = self.test_files_dir / "test.txt"
        test_file.write_text("This is a test file")
        
        try:
            # Use fetch API to send a request with invalid content type
            result = page.evaluate("""async (url, filePath) => {
                const formData = new FormData();
                const file = new File(['dummy'], 'test.txt', { type: 'application/octet-stream' });
                formData.append('file', file);
                
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData,
                    // Intentionally omit Content-Type to trigger browser to set it with boundary
                });
                
                return {
                    status: response.status,
                    body: await response.text()
                };
            }""", f"{self.base_url}/api/upload", str(test_file))
            
            # Should either reject with 400 or process with correct content type
            assert result.status in (200, 400), \
                f"Unexpected status code: {result.status}"
                
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_missing_required_fields(self, page: Page):
        """Test form submission with missing required fields"""
        # Test login form with missing fields
        page.goto(f"{self.base_url}/login")
        
        # Submit empty form
        with page.expect_response(lambda response: 
            response.status == 400  # Bad Request
        ) as response_info:
            page.click("button[type=submit]")
        
        # Check for validation errors
        error_messages = page.locator(".error-message")
        expect(error_messages).to_have_count(2)  # Username and password required
        
        # Fill only username
        page.fill("input[name=username]", "testuser")
        page.click("button[type=submit]")
        
        # Should still show password error
        error_messages = page.locator(".error-message")
        expect(error_messages).to_have_count(1)
        expect(error_messages).to_contain_text("password")
    
    def test_session_timeout(self, page: Page):
        """Test session timeout handling"""
        # Login first
        page.goto(f"{self.base_url}/login")
        page.fill("input[name=username]", "testuser")
        page.fill("input[name=password]", "testpass123")
        page.click("button[type=submit]")
        
        # Wait for dashboard to load
        page.wait_for_selector("#dashboard")
        
        # Simulate session timeout (adjust based on your session timeout)
        page.evaluate("""() => {
            // Clear session storage
            sessionStorage.clear();
            // Clear local storage if needed
            localStorage.clear();
            // Remove auth cookies
            document.cookie.split(";").forEach(cookie => {
                document.cookie = cookie.trim().split("=")[0] + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
            });
        }""")
        
        # Try to access a protected route
        page.goto(f"{self.base_url}/dashboard")
        
        # Should be redirected to login page
        expect(page).to_have_url(f"{self.base_url}/login?returnUrl=%2Fdashboard")
    
    def test_concurrent_sessions(self, page: Page, context):
        """Test handling of concurrent sessions"""
        # Login in first tab
        page.goto(f"{self.base_url}/login")
        page.fill("input[name=username]", "testuser")
        page.fill("input[name=password]", "testpass123")
        page.click("button[type=submit]")
        
        # Wait for dashboard to load
        page.wait_for_selector("#dashboard")
        
        # Open a new tab and try to login with same credentials
        new_page = context.new_page()
        try:
            new_page.goto(f"{self.base_url}/login")
            new_page.fill("input[name=username]", "testuser")
            new_page.fill("input[name=password]", "testpass123")
            
            # This should either:
            # 1. Log out the first session, or
            # 2. Prevent the second login, or
            # 3. Allow both (if your app supports it)
            
            # For this test, we'll just verify the app handles it gracefully
            with new_page.expect_navigation():
                new_page.click("button[type=submit]")
            
            # Verify we're either logged in or shown an appropriate message
            assert "dashboard" in new_page.url or "already logged in" in new_page.content().lower()
            
        finally:
            new_page.close()
    
    def test_error_pages(self, page: Page):
        """Test custom error pages"""
        # Test 404 page
        page.goto(f"{self.base_url}/nonexistent-page")
        
        # Should show custom 404 page
        error_title = page.locator("h1")
        expect(error_title).to_contain_text("404")
        
        # Test 500 error (simulate by hitting an error endpoint if available)
        page.goto(f"{self.base_url}/api/simulate-error")
        
        # Should show custom 500 page or return 500 status
        assert page.status_code in (200, 500), \
            f"Expected 200 (with error page) or 500, got {page.status_code}"
