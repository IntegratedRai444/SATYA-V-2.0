"""
UI Component Tests using Playwright
"""
import pytest
from playwright.sync_api import Page, expect

class TestUIComponents:
    """Test UI components and interactions"""
    
    def test_navigation(self, page: Page, base_url):
        """Test main navigation"""
        page.goto(f"{base_url}")
        
        # Test navigation links
        nav_links = [
            ("Home", "/"),
            ("Analyze", "/analyze"),
            ("History", "/history"),
            ("API", "/api-docs")
        ]
        
        for text, href in nav_links:
            link = page.get_by_role("link", name=text)
            expect(link).to_be_visible()
            assert href in link.get_attribute("href")
    
    def test_dark_mode_toggle(self, page: Page, base_url):
        """Test dark/light mode toggle"""
        page.goto(f"{base_url}")
        
        # Get initial theme
        initial_theme = page.evaluate('document.documentElement.getAttribute("data-theme")')
        
        # Toggle theme
        theme_toggle = page.get_by_role("button", name="Toggle theme")
        theme_toggle.click()
        
        # Verify theme changed
        new_theme = page.evaluate('document.documentElement.getAttribute("data-theme")')
        assert new_theme != initial_theme

class TestFormValidations:
    """Test form validations and submissions"""
    
    def test_required_fields(self, page: Page, base_url):
        """Test form validation for required fields"""
        page.goto(f"{base_url}/analyze")
        
        # Submit empty form
        submit_button = page.get_by_role("button", name={"exact": True}, value="Analyze")
        submit_button.click()
        
        # Check for validation errors
        error_messages = page.locator(".error-message")
        expect(error_messages).to_have_count(1)
        expect(error_messages.first).to_contain_text("Please select a file")
    
    def test_file_upload_validation(self, page: Page, base_url):
        """Test file upload validation"""
        page.goto(f"{base_url}/analyze")
        
        # Upload invalid file type
        with page.expect_file_chooser() as fc_info:
            page.get_by_text("Choose File").click()
        file_chooser = fc_info.value
        
        # Create a temporary invalid file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is not an image")
            tmp_path = tmp.name
        
        try:
            file_chooser.set_files(tmp_path)
            
            # Check for error message
            error_message = page.locator(".file-error")
            expect(error_message).to_contain_text("Unsupported file type")
            
            # Verify submit is disabled
            submit_button = page.get_by_role("button", name={"exact": True}, value="Analyze")
            expect(submit_button).to_be_disabled()
        finally:
            import os
            os.unlink(tmp_path)
    
    def test_successful_analysis_flow(self, page: Page, base_url):
        """Test complete analysis flow with valid file"""
        page.goto(f"{base_url}/analyze")
        
        # Mock the API response
        page.route("**/api/analyze/image", lambda route: route.fulfill(
            status=202,
            content_type="application/json",
            body='{"job_id": "test-job-123"}'
        ))
        
        # Create a test image file
        from PIL import Image
        import io
        
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Upload the test image
        with page.expect_file_chooser() as fc_info:
            page.get_by_text("Choose File").click()
        file_chooser = fc_info.value
        file_chooser.set_files({
            "name": "test.png",
            "mimeType": "image/png",
            "buffer": img_byte_arr
        })
        
        # Submit the form
        submit_button = page.get_by_role("button", name={"exact": True}, value="Analyze")
        submit_button.click()
        
        # Verify loading state
        loading_indicator = page.locator(".loading-indicator")
        expect(loading_indicator).to_be_visible()
        
        # Mock the results page
        page.route("**/api/analyze/test-job-123/results", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='''{
                "status": "completed",
                "results": {
                    "is_fake": false,
                    "confidence": 0.95,
                    "details": {}
                }
            }'''
        ))
        
        # Wait for results
        results_section = page.locator(".analysis-results")
        expect(results_section).to_be_visible()
        
        # Verify result content
        confidence_text = page.locator(".confidence-value")
        expect(confidence_text).to_contain_text("95%")
        
        result_status = page.locator(".result-status")
        expect(result_status).to_contain_text("Authentic")
