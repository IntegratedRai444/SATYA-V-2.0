"""
Accessibility Testing Configuration
"""
import pytest
from playwright.sync_api import Page, expect
from axe_playwright_python.sync_playwright import Axe
import json
from pathlib import Path

# Configure output directories
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def axe():
    """Create an Axe accessibility testing instance"""
    return Axe()

@pytest.fixture(autouse=True)
def setup_accessibility(page: Page):
    """Setup for accessibility tests"""
    # Set the viewport to a consistent size
    page.set_viewport_size({"width": 1280, "height": 1024})
    
    # Add a custom error handler for accessibility violations
    def handle_accessibility_violations(violations, page_title):
        if violations:
            # Save violations to a file
            output_file = OUTPUT_DIR / f"{page_title}_violations.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(violations, f, indent=2)
            
            # Print a summary
            print(f"\nFound {len(violations)} accessibility violations on {page_title}:")
            for i, violation in enumerate(violations, 1):
                print(f"{i}. {violation['id']}: {violation['description']}")
                print(f"   Impact: {violation['impact']}")
                print(f"   Help: {violation['helpUrl']}")
                print(f"   Elements: {len(violation['nodes'])}")
    
    # Add the handler to the page object
    page.accessibility_violations = []
    page.check_accessibility = lambda: handle_accessibility_violations(
        page.accessibility_violations, 
        page.title().lower().replace(' ', '_') or 'page'
    )
    
    yield
    
    # After each test, check for accessibility violations
    if hasattr(page, 'accessibility_violations') and page.accessibility_violations:
        page.check_accessibility()

def pytest_addoption(parser):
    """Add command line options for accessibility testing"""
    parser.addoption(
        "--accessibility",
        action="store_true",
        default=False,
        help="Run accessibility tests"
    )
    parser.addoption(
        "--a11y-rules",
        action="store",
        default="wcag2a,wcag2aa",
        help="Comma-separated list of accessibility rules to check"
    )

def pytest_configure(config):
    """Configure pytest for accessibility testing"""
    config.addinivalue_line(
        "markers",
        "accessibility: mark test as an accessibility test"
    )

def pytest_collection_modifyitems(config, items):
    """Skip accessibility tests if not explicitly requested"""
    if not config.getoption("--accessibility"):
        skip_accessibility = pytest.mark.skip(
            reason="Need --accessibility option to run accessibility tests"
        )
        for item in items:
            if "accessibility" in item.keywords:
                item.add_marker(skip_accessibility)
