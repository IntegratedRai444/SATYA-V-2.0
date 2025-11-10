"""
API Interaction Tests using Playwright
"""
import json
import pytest
from playwright.sync_api import Page, expect

class TestAPIIntegrations:
    """Test API interactions from the browser"""
    
    def test_authentication_flow(self, page: Page, base_url: str):
        """Test login/logout flow with API calls"""
        # Mock the login API response
        page.route(
            "**/api/auth/login",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"token": "test-token-123", "user": {"username": "testuser"}})
            )
        )
        
        # Navigate to login page
        page.goto(f"{base_url}/login")
        
        # Fill and submit login form
        page.fill("input[name=username]", "testuser")
        page.fill("input[name=password]", "testpass123")
        
        # Mock the user data response after successful login
        page.route(
            "**/api/user/me",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"username": "testuser", "email": "test@example.com"})
            )
        )
        
        # Submit the form
        with page.expect_navigation():
            page.click("button[type=submit]")
        
        # Verify successful login
        expect(page).to_have_url(f"{base_url}/dashboard")
        
        # Test logout
        page.route(
            "**/api/auth/logout",
            lambda route: route.fulfill(status=200, body="")
        )
        
        # Click logout button
        page.click("button#logout")
        
        # Verify redirect to login page
        expect(page).to_have_url(f"{base_url}/login")

    def test_protected_routes(self, page: Page, base_url: str):
        """Test that protected routes redirect to login when not authenticated"""
        protected_routes = [
            "/dashboard",
            "/profile",
            "/settings"
        ]
        
        for route in protected_routes:
            page.goto(f"{base_url}{route}")
            expect(page).to_have_url(f"{base_url}/login?returnUrl={route}")

    def test_api_error_handling(self, page: Page, base_url: str):
        """Test handling of API errors"""
        # Test 401 Unauthorized
        page.route(
            "**/api/protected-route",
            lambda route: route.fulfill(status=401, body="Unauthorized")
        )
        
        # Try to access protected route
        page.goto(f"{base_url}/protected-route")
        
        # Verify error message is shown
        error_message = page.locator(".error-message")
        expect(error_message).to_contain_text("Please log in")
        
        # Test 500 Server Error
        page.route(
            "**/api/failing-endpoint",
            lambda route: route.fulfill(status=500, body="Internal Server Error")
        )
        
        # Trigger the failing request
        page.goto(f"{base_url}/trigger-error")
        
        # Verify error handling
        error_message = page.locator(".error-message")
        expect(error_message).to_contain_text("Something went wrong")

class TestFileUploads:
    """Test file upload functionality"""
    
    def test_image_upload(self, page: Page, base_url: str):
        """Test successful image upload"""
        # Mock the upload endpoint
        page.route(
            "**/api/analyze/image",
            lambda route: route.fulfill(
                status=202,
                content_type="application/json",
                body=json.dumps({"job_id": "test-job-123"})
            )
        )
        
        # Navigate to upload page
        page.goto(f"{base_url}/upload")
        
        # Create a test file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"test image content")
            file_path = tmp.name
        
        try:
            # Upload the file
            with page.expect_file_chooser() as fc_info:
                page.click("input[type=file]")
            file_chooser = fc_info.value
            file_chooser.set_files(file_path)
            
            # Submit the form
            with page.expect_response("**/api/analyze/image") as response_info:
                page.click("button[type=submit]")
            
            response = response_info.value
            assert response.status == 202
            
            # Verify success message
            success_message = page.locator(".success-message")
            expect(success_message).to_be_visible()
            
        finally:
            # Clean up
            import os
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_large_file_upload(self, page: Page, base_url: str):
        """Test handling of large file uploads"""
        # Create a large file (5MB)
        import tempfile
        large_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        large_file.write(b"0" * (5 * 1024 * 1024))  # 5MB
        large_file.close()
        
        try:
            # Mock the file size check
            page.route(
                "**/api/analyze/image",
                lambda route: route.fulfill(status=413, body="File too large")
            )
            
            # Navigate to upload page
            page.goto(f"{base_url}/upload")
            
            # Upload the large file
            with page.expect_file_chooser() as fc_info:
                page.click("input[type=file]")
            file_chooser = fc_info.value
            file_chooser.set_files(large_file.name)
            
            # Submit the form
            page.click("button[type=submit]")
            
            # Verify error message
            error_message = page.locator(".error-message")
            expect(error_message).to_contain_text("too large")
            
        finally:
            # Clean up
            import os
            if os.path.exists(large_file.name):
                os.unlink(large_file.name)
