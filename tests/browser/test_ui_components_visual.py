"""
Visual Regression Tests for UI Components
"""
import os
import pytest
from pathlib import Path
from typing import Dict, Any
from playwright.sync_api import Page, expect
from PIL import Image, ImageChops, ImageFilter
import numpy as np
import hashlib

# Threshold for considering images different (0-1)
PIXEL_MATCH_THRESHOLD = 0.1
# Threshold for considering images similar (0-1)
SIMILARITY_THRESHOLD = 0.9

class TestUIVisualRegression:
    """Visual regression tests for UI components"""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.screenshots_dir = Path(__file__).parent / "screenshots"
        self.baseline_dir = self.screenshots_dir / "baseline"
        self.diff_dir = self.screenshots_dir / "diff"
        self.current_dir = self.screenshots_dir / "current"
        
        # Create directories if they don't exist
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)
    
    def take_screenshot(self, name: str, full_page: bool = False, element_selector: str = None) -> Path:
        """Take a screenshot and save it to the screenshots directory"""
        # Ensure the page is stable before taking screenshot
        self.page.wait_for_load_state("networkidle")
        
        # Handle element-specific screenshots
        if element_selector:
            element = self.page.locator(element_selector).first
            element.scroll_into_view_if_needed()
            element.screenshot(path=str(self.current_dir / f"{name}.png"))
            return self.current_dir / f"{name}.png"
        
        # Full page or viewport screenshot
        screenshot_path = self.current_dir / f"{name}.png"
        self.page.screenshot(path=str(screenshot_path), full_page=full_page)
        return screenshot_path
    
    def calculate_similarity(self, img1_path: Path, img2_path: Path) -> float:
        """Calculate the similarity between two images (0-1)"""
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Ensure images have the same size
            if img1.size != img2.size:
                return 0.0
                
            # Calculate the difference between the two images
            diff = ImageChops.difference(img1, img2)
            
            # Convert to grayscale and calculate mean difference
            diff_gray = diff.convert('L')
            diff_array = np.array(diff_gray)
            
            # Calculate similarity (1 - normalized difference)
            similarity = 1.0 - (np.mean(diff_array) / 255.0)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"Error calculating image similarity: {e}")
            return 0.0
    
    def save_diff_image(self, img1_path: Path, img2_path: Path, diff_path: Path):
        """Generate and save a visual diff image"""
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Ensure images have the same size
            if img1.size != img2.size:
                print("Images have different sizes, cannot create diff")
                return
                
            # Calculate difference
            diff = ImageChops.difference(img1, img2)
            
            # Enhance the difference for better visibility
            diff = diff.convert('L')
            diff = diff.point(lambda x: 0 if x < 25 else 255)
            
            # Save the diff image
            diff.save(diff_path)
            
        except Exception as e:
            print(f"Error creating diff image: {e}")
    
    def test_navigation_menu(self, page: Page):
        """Test that the navigation menu renders correctly"""
        page.goto(f"{self.base_url}")
        
        # Wait for navigation to be visible
        nav_selector = "nav, [role='navigation'], .navbar, .nav"
        page.wait_for_selector(nav_selector, state="visible")
        
        # Take screenshot of the navigation
        screenshot_path = self.take_screenshot("navigation_menu", element_selector=nav_selector)
        baseline_path = self.baseline_dir / "navigation_menu.png"
        diff_path = self.diff_dir / "navigation_menu_diff.png"
        
        # If baseline doesn't exist, save current as baseline
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.rename(baseline_path)
            pytest.skip(f"Created baseline image at {baseline_path}")
        
        # Calculate similarity with baseline
        similarity = self.calculate_similarity(screenshot_path, baseline_path)
        
        # Save diff if similarity is below threshold
        if similarity < SIMILARITY_THRESHOLD:
            self.save_diff_image(screenshot_path, baseline_path, diff_path)
            
        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Navigation menu visual regression detected. Similarity: {similarity*100:.2f}%"
    
    def test_form_elements(self, page: Page):
        """Test that form elements render correctly"""
        page.goto(f"{self.base_url}/login")
        
        # Wait for form to be visible
        form_selector = "form, [role='form']"
        page.wait_for_selector(form_selector, state="visible")
        
        # Take screenshot of the form
        screenshot_path = self.take_screenshot("login_form", element_selector=form_selector)
        baseline_path = self.baseline_dir / "login_form.png"
        diff_path = self.diff_dir / "login_form_diff.png"
        
        # If baseline doesn't exist, save current as baseline
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.rename(baseline_path)
            pytest.skip(f"Created baseline image at {baseline_path}")
        
        # Calculate similarity with baseline
        similarity = self.calculate_similarity(screenshot_path, baseline_path)
        
        # Save diff if similarity is below threshold
        if similarity < SIMILARITY_THRESHOLD:
            self.save_diff_image(screenshot_path, baseline_path, diff_path)
            
        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Login form visual regression detected. Similarity: {similarity*100:.2f}%"
    
    def test_buttons(self, page: Page):
        """Test that buttons render correctly"""
        page.goto(f"{self.base_url}")
        
        # Wait for buttons to be visible
        button_selector = "button, [role='button'], .btn, .button"
        page.wait_for_selector(button_selector, state="visible")
        
        # Take screenshot of buttons
        screenshot_path = self.take_screenshot("buttons", element_selector=button_selector)
        baseline_path = self.baseline_dir / "buttons.png"
        diff_path = self.diff_dir / "buttons_diff.png"
        
        # If baseline doesn't exist, save current as baseline
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.rename(baseline_path)
            pytest.skip(f"Created baseline image at {baseline_path}")
        
        # Calculate similarity with baseline
        similarity = self.calculate_similarity(screenshot_path, baseline_path)
        
        # Save diff if similarity is below threshold
        if similarity < SIMILARITY_THRESHOLD:
            self.save_diff_image(screenshot_path, baseline_path, diff_path)
            
        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Button visual regression detected. Similarity: {similarity*100:.2f}%"
    
    def test_mobile_view(self, page: Page):
        """Test that the mobile view renders correctly"""
        # Set mobile viewport
        mobile_viewport = {"width": 375, "height": 812, "isMobile": True}
        page.set_viewport_size(mobile_viewport)
        
        page.goto(f"{self.base_url}")
        
        # Wait for page to load
        page.wait_for_selector("body", state="visible")
        
        # Take full page screenshot
        screenshot_path = self.take_screenshot("mobile_homepage", full_page=True)
        baseline_path = self.baseline_dir / "mobile_homepage.png"
        diff_path = self.diff_dir / "mobile_homepage_diff.png"
        
        # If baseline doesn't exist, save current as baseline
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.rename(baseline_path)
            pytest.skip(f"Created baseline image at {baseline_path}")
        
        # Calculate similarity with baseline
        similarity = self.calculate_similarity(screenshot_path, baseline_path)
        
        # Save diff if similarity is below threshold
        if similarity < SIMILARITY_THRESHOLD:
            self.save_diff_image(screenshot_path, baseline_path, diff_path)
            
        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Mobile view visual regression detected. Similarity: {similarity*100:.2f}%"
    
    def test_dark_mode(self, page: Page):
        """Test that dark mode renders correctly"""
        # Enable dark mode via localStorage
        page.goto(f"{self.base_url}")
        page.evaluate("""() => {
            localStorage.setItem('theme', 'dark');
            document.documentElement.classList.add('dark');
        }""")
        
        # Reload to apply dark mode
        page.reload()
        
        # Wait for page to load
        page.wait_for_selector("body", state="visible")
        
        # Take screenshot
        screenshot_path = self.take_screenshot("dark_mode", full_page=True)
        baseline_path = self.baseline_dir / "dark_mode.png"
        diff_path = self.diff_dir / "dark_mode_diff.png"
        
        # If baseline doesn't exist, save current as baseline
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.rename(baseline_path)
            pytest.skip(f"Created baseline image at {baseline_path}")
        
        # Calculate similarity with baseline
        similarity = self.calculate_similarity(screenshot_path, baseline_path)
        
        # Save diff if similarity is below threshold
        if similarity < SIMILARITY_THRESHOLD:
            self.save_diff_image(screenshot_path, baseline_path, diff_path)
            
        assert similarity >= SIMILARITY_THRESHOLD, \
            f"Dark mode visual regression detected. Similarity: {similarity*100:.2f}%"
        
        # Clean up - reset to light mode
        page.evaluate("""() => {
            localStorage.setItem('theme', 'light');
            document.documentElement.classList.remove('dark');
        }""")
