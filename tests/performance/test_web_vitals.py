"""
Performance Testing with Playwright and Lighthouse
"""
import json
import pytest
from pathlib import Path
from playwright.sync_api import Page, expect
from typing import Dict, Any

# Performance thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS = {
    "load_time": 3000,  # Time to load the page
    "first_contentful_paint": 2000,  # FCP
    "largest_contentful_paint": 2500,  # LCP
    "cumulative_layout_shift": 0.1,  # CLS
    "first_input_delay": 100,  # FID
    "time_to_interactive": 3500,  # TTI
}

class TestWebPerformance:
    """Performance testing for web vitals and page load metrics"""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        self.page = page
        self.metrics: Dict[str, Any] = {}
    
    def capture_metrics(self, page: Page, name: str):
        """Capture performance metrics for the current page"""
        # Get Web Vitals using JavaScript
        metrics = page.evaluate("""() => {
            const {
                fetchStart,
                domContentLoadedEventEnd,
                loadEventEnd,
                firstPaint,
                firstContentfulPaint,
                largestContentfulPaint,
                cumulativeLayoutShift,
                firstInputDelay,
            } = window.performance.getEntriesByType('navigation')[0];
            
            return {
                load_time: loadEventEnd - fetchStart,
                dom_content_loaded: domContentLoadedEventEnd - fetchStart,
                first_paint: firstPaint,
                first_contentful_paint: firstContentfulPaint,
                largest_contentful_paint: largestContentfulPaint,
                cumulative_layout_shift: cumulativeLayoutShift,
                first_input_delay: firstInputDelay,
            };
        }""")
        
        # Store metrics
        self.metrics[name] = metrics
        return metrics
    
    def assert_metrics(self, metrics: Dict[str, float]):
        """Assert that metrics meet performance thresholds"""
        for metric, threshold in PERFORMANCE_THRESHOLDS.items():
            value = metrics.get(metric, 0)
            assert value <= threshold, f"{metric} of {value}ms exceeds threshold of {threshold}ms"
    
    def test_homepage_performance(self, page: Page, base_url: str):
        """Test homepage load performance"""
        # Enable performance monitoring
        page.goto(f"{base_url}", wait_until="networkidle")
        
        # Wait for critical elements
        page.wait_for_selector("main", state="visible")
        
        # Capture metrics
        metrics = self.capture_metrics(page, "homepage")
        
        # Assert performance
        self.assert_metrics(metrics)
        
        # Log metrics
        print("\nHomepage Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}ms")
    
    def test_login_performance(self, page: Page, base_url: str):
        """Test login page performance"""
        # Navigate to login page
        page.goto(f"{base_url}/login", wait_until="networkidle")
        
        # Fill login form
        page.fill("input[name=username]", "testuser")
        page.fill("input[name=password]", "testpass123")
        
        # Start performance measurement
        page.evaluate("window.performance.mark('login-start')")
        
        # Submit form
        page.click("button[type=submit]")
        
        # Wait for navigation
        page.wait_for_selector("#dashboard", state="visible")
        
        # Measure login time
        login_time = page.evaluate("""() => {
            window.performance.mark('login-end');
            window.performance.measure('login', 'login-start', 'login-end');
            const measure = window.performance.getEntriesByName('login')[0];
            return measure.duration;
        }""")
        
        # Store login time
        metrics = {"login_time": login_time}
        self.metrics["login"] = metrics
        
        # Assert performance
        assert login_time <= 2000, f"Login took {login_time}ms, expected <= 2000ms"
        
        # Log metrics
        print(f"\nLogin Performance: {login_time:.2f}ms")
    
    def test_lighthouse_audit(self, page: Page, base_url: str):
        """Run Lighthouse audit for performance"""
        try:
            # This requires the Playwright Lighthouse plugin
            from playwright_lighthouse import lighthouse
            
            # Run Lighthouse audit
            lighthouse_audit = lighthouse(
                page,
                port=9222,
                output="html",
                output_path=str(Path("lighthouse") / "report.html"),
                reports_config={
                    "formats": {
                        "html": True,
                        "json": True,
                        "csv": False
                    },
                },
                disable_comments=True,
                disable_emulated_form_factor=True,
                skip_audits=["uses-http2"],
            )
            
            # Get scores
            scores = lighthouse_audit.get("score", {})
            
            # Assert minimum scores (0-1 scale)
            assert scores.get("performance", 0) >= 0.8, "Performance score too low"
            assert scores.get("accessibility", 0) >= 0.9, "Accessibility score too low"
            assert scores.get("best-practices", 0) >= 0.9, "Best practices score too low"
            assert scores.get("seo", 0) >= 0.9, "SEO score too low"
            
            # Log scores
            print("\nLighthouse Scores:")
            for category, score in scores.items():
                print(f"{category}: {score*100:.0f}")
                
        except ImportError:
            pytest.skip("Playwright Lighthouse plugin not installed")
