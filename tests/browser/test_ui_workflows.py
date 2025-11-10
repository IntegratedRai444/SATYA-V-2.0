"""
Cross-browser UI tests for SatyaAI
"""
import pytest
import time
from pathlib import Path

class TestCrossBrowserUI:
    """Cross-browser UI test cases"""
    
    def test_homepage_loads(self, browser, browser_config):
        """Test that homepage loads correctly"""
        browser.get(browser_config["base_url"])
        
        # Check for key elements
        assert "SatyaAI" in browser.title
        assert browser.find_element_by_css_selector("header h1").is_displayed()
        assert browser.find_element_by_css_selector("nav").is_displayed()
    
    def test_login_workflow(self, browser, browser_config):
        """Test login workflow"""
        browser.get(f"{browser_config['base_url']}/login")
        
        # Fill and submit login form
        username = browser.find_element_by_id("username")
        password = browser.find_element_by_id("password")
        submit = browser.find_element_by_css_selector("button[type='submit']")
        
        username.send_keys("testuser")
        password.send_keys("testpass123")
        submit.click()
        
        # Verify successful login
        assert "dashboard" in browser.current_url
        assert browser.find_element_by_id("user-greeting").is_displayed()
    
    def test_file_upload_workflow(self, login):
        """Test file upload workflow"""
        browser = login
        
        # Navigate to upload page
        upload_button = browser.find_element_by_css_selector("a[href='/upload']")
        upload_button.click()
        
        # Upload test file
        test_file = str(Path(__file__).parent.parent / "test_files" / "test_face.jpg")
        file_input = browser.find_element_by_css_selector("input[type='file']")
        file_input.send_keys(test_file)
        
        # Submit form
        submit_button = browser.find_element_by_css_selector("button[type='submit']")
        submit_button.click()
        
        # Verify upload success
        assert "analysis" in browser.current_url
        assert browser.find_element_by_id("analysis-progress").is_displayed()
    
    @pytest.mark.parametrize("browser_type", ["chrome", "firefox"])
    def test_browser_compatibility(self, browser, browser_type):
        """Test basic functionality across different browsers"""
        if browser.name.lower() != browser_type:
            pytest.skip(f"Skipping {browser_type} test")
            
        browser.get("https://www.whatsmybrowser.org/")
        time.sleep(2)  # Wait for page to load
        
        # Basic browser test
        assert browser.title == "What's my browser?"
        
        # JavaScript test
        js_result = browser.execute_script("return 1 + 1")
        assert js_result == 2, "JavaScript execution failed"
