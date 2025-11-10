"""
Configuration for browser-based tests with Playwright
"""
import os
import pytest
from typing import Generator
from pathlib import Path
from playwright.sync_api import Page, Browser, BrowserContext, expect

# Test configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:3000")
DEFAULT_TIMEOUT = 10000  # milliseconds
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
TEST_FILES_DIR = Path(__file__).parent / "test_files"

# Create directories if they don't exist
SCREENSHOT_DIR.mkdir(exist_ok=True)
TEST_FILES_DIR.mkdir(exist_ok=True)

@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Configure browser launch arguments"""
    return {
        **browser_type_launch_args,
        "headless": os.getenv("HEADLESS", "true").lower() == "true",
        "slow_mo": int(os.getenv("SLOW_MO", "0")),
    }

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context"""
    return {
        **browser_context_args,
        "viewport": {
            "width": 1920,
            "height": 1080,
        },
        "ignore_https_errors": True,
    }

@pytest.fixture(scope="function")
def context(browser: Browser, browser_context_args) -> Generator[BrowserContext, None, None]:
    """Create a new browser context for each test"""
    context = browser.new_context(**browser_context_args)
    
    # Set default timeout
    context.set_default_timeout(DEFAULT_TIMEOUT)
    
    # Start tracing
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    
    yield context
    
    # Save trace on test failure
    if hasattr(pytest, "test_failed") and pytest.test_failed:
        trace_path = SCREENSHOT_DIR / f"trace-{pytest.test_name}.zip"
        context.tracing.stop(path=trace_path)
    
    # Close the context
    context.close()

@pytest.fixture(scope="function")
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create a new page for each test"""
    page = context.new_page()
    
    # Set up error handling
    def handle_page_error(error):
        print(f"Page error: {error}")
    
    page.on("pageerror", handle_page_error)
    
    yield page
    
    # Take screenshot on test failure
    if hasattr(pytest, "test_failed") and pytest.test_failed:
        screenshot_path = SCREENSHOT_DIR / f"{pytest.test_name}.png"
        page.screenshot(path=screenshot_path)

@pytest.fixture(scope="function")
def base_url() -> str:
    """Get the base URL for tests"""
    return BASE_URL

@pytest.fixture(scope="function")
def login(page: Page, base_url: str):
    """Login helper function"""
    def _login(username=None, password=None):
        username = username or os.getenv("TEST_USERNAME", "testuser")
        password = password or os.getenv("TEST_PASSWORD", "testpass123")
        
        page.goto(f"{base_url}/login")
        
        # Fill login form
        page.fill("input[name=username]", username)
        page.fill("input[name=password]", password)
        page.click("button[type=submit]")
        
        # Wait for navigation
        page.wait_for_selector("#dashboard", state="visible")
        
        return page
    
    return _login

@pytest.fixture(autouse=True)
def test_setup_teardown(request):
    """Setup and teardown for each test"""
    # Store test name for screenshots
    pytest.test_name = request.node.name.replace(" ", "_")
    pytest.test_failed = False
    
    yield
    
    # Mark test as failed if it raised an exception
    if request.node.rep_call.failed if hasattr(request.node, 'rep_call') else False:
        pytest.test_failed = True
