"""
Visual Regression Tests using Playwright and pixelmatch
"""
import os
import pytest
from pathlib import Path
from playwright.sync_api import Page, expect
from pixelmatch.contrib.PIL import pixelmatch
from PIL import Image
import numpy as np

# Threshold for considering images different (0-1)
PIXEL_MATCH_THRESHOLD = 0.1

class TestVisualRegression:
    """Visual regression tests for UI components"""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.screenshots_dir = Path(__file__).parent / "screenshots"
        self.baseline_dir = self.screenshots_dir / "baseline"
        self.diff_dir = self.screenshots_dir / "diff"
        
        # Create directories if they don't exist
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.diff_dir.mkdir(parents=True, exist_ok=True)
    
    def take_screenshot(self, name: str, full_page: bool = False) -> Path:
        """Take a screenshot and save it to the screenshots directory"""
        screenshot_path = self.screenshots_dir / f"{name}.png"
        self.page.screenshot(path=str(screenshot_path), full_page=full_page)
        return screenshot_path
    
    def compare_images(self, actual_path: Path, expected_path: Path, diff_path: Path) -> float:
        """Compare two images and return the difference ratio"""
        if not expected_path.exists():
            # If no baseline exists, save the current screenshot as baseline
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            actual_path.rename(expected_path)
            return 0.0
        
        # Load images
        img_actual = Image.open(actual_path)
        img_expected = Image.open(expected_path)
        
        # Ensure images have the same size
        if img_actual.size != img_expected.size:
            return 1.0  # 100% different if sizes don't match
        
        # Convert to RGBA if needed
        if img_actual.mode != 'RGBA':
            img_actual = img_actual.convert('RGBA')
        if img_expected.mode != 'RGBA':
            img_expected = img_expected.convert('RGBA')
        
        # Create diff image
        diff = Image.new('RGBA', img_actual.size)
        
        # Compare images
        mismatch = pixelmatch(
            img_actual, img_expected, 
            diff, threshold=0.1
        )
        
        # Calculate difference ratio
        diff_ratio = mismatch / (img_actual.width * img_actual.height)
        
        # Save diff if there are differences
        if diff_ratio > 0:
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff.save(diff_path)
        
        return diff_ratio
    
    def test_homepage_layout(self, page: Page, base_url: str):
        """Test homepage layout matches baseline"""
        # Navigate to homepage
        page.goto(f"{base_url}")
        
        # Wait for page to load
        page.wait_for_selector("main", state="visible")
        
        # Take screenshot
        screenshot_path = self.take_screenshot("homepage")
        baseline_path = self.baseline_dir / "homepage.png"
        diff_path = self.diff_dir / "homepage_diff.png"
        
        # Compare with baseline
        diff_ratio = self.compare_images(screenshot_path, baseline_path, diff_path)
        
        # Fail if difference is above threshold
        assert diff_ratio <= PIXEL_MATCH_THRESHOLD, \
            f"Visual regression detected: {diff_ratio*100:.2f}% difference from baseline"
    
    def test_login_page_layout(self, page: Page, base_url: str):
        """Test login page layout matches baseline"""
        # Navigate to login page
        page.goto(f"{base_url}/login")
        
        # Wait for page to load
        page.wait_for_selector("form", state="visible")
        
        # Take screenshot
        screenshot_path = self.take_screenshot("login_page")
        baseline_path = self.baseline_dir / "login_page.png"
        diff_path = self.diff_dir / "login_page_diff.png"
        
        # Compare with baseline
        diff_ratio = self.compare_images(screenshot_path, baseline_path, diff_path)
        
        # Fail if difference is above threshold
        assert diff_ratio <= PIXEL_MATCH_THRESHOLD, \
            f"Visual regression detected: {diff_ratio*100:.2f}% difference from baseline"
    
    def test_dashboard_layout(self, page: Page, base_url: str, login):
        """Test dashboard layout matches baseline"""
        # Login first
        login()
        
        # Wait for dashboard to load
        page.wait_for_selector("#dashboard", state="visible")
        
        # Take screenshot
        screenshot_path = self.take_screenshot("dashboard")
        baseline_path = self.baseline_dir / "dashboard.png"
        diff_path = self.diff_dir / "dashboard_diff.png"
        
        # Compare with baseline
        diff_ratio = self.compare_images(screenshot_path, baseline_path, diff_path)
        
        # Fail if difference is above threshold
        assert diff_ratio <= PIXEL_MATCH_THRESHOLD, \
            f"Visual regression detected: {diff_ratio*100:.2f}% difference from baseline"
