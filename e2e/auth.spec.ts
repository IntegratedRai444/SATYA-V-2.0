import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should display login page correctly', async ({ page }) => {
    await page.goto('/login');
    
    // Check page title
    await expect(page).toHaveTitle(/SATYA AI/);
    
    // Check login form elements
    await expect(page.locator('input[name="email"]')).toBeVisible();
    await expect(page.locator('input[name="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
    
    // Check navigation links
    await expect(page.locator('text=Create an account')).toBeVisible();
    await expect(page.locator('text=Forgot password?')).toBeVisible();
  });

  test('should show validation errors for empty form', async ({ page }) => {
    await page.goto('/login');
    
    // Try to submit empty form
    await page.locator('button[type="submit"]').click();
    
    // Check for validation messages
    await expect(page.locator('text=Email is required')).toBeVisible();
    await expect(page.locator('text=Password is required')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Fill with invalid credentials
    await page.fill('input[name="email"]', 'invalid@test.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.locator('button[type="submit"]').click();
    
    // Check for error message
    await expect(page.locator('text=Invalid credentials')).toBeVisible({ timeout: 10000 });
  });

  test('should navigate to registration page', async ({ page }) => {
    await page.goto('/login');
    
    // Click registration link
    await page.click('text=Create an account');
    
    // Should be on registration page
    await expect(page).toHaveURL(/.*register/);
    await expect(page.locator('h1')).toContainText('Create Account');
  });

  test('should register new user successfully', async ({ page }) => {
    const timestamp = Date.now();
    const email = `testuser${timestamp}@test.com`;
    
    await page.goto('/register');
    
    // Fill registration form
    await page.fill('input[name="fullName"]', `Test User ${timestamp}`);
    await page.fill('input[name="email"]', email);
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.fill('input[name="confirmPassword"]', 'TestPass123!');
    
    // Submit form
    await page.locator('button[type="submit"]').click();
    
    // Should redirect to email verification or dashboard
    await expect(page.locator('text=Registration successful')).toBeVisible({ timeout: 15000 });
  });

  test('should handle password confirmation mismatch', async ({ page }) => {
    await page.goto('/register');
    
    // Fill form with mismatched passwords
    await page.fill('input[name="fullName"]', 'Test User');
    await page.fill('input[name="email"]', 'test@test.com');
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.fill('input[name="confirmPassword"]', 'DifferentPass123!');
    
    // Submit form
    await page.locator('button[type="submit"]').click();
    
    // Should show error
    await expect(page.locator('text=Passwords do not match')).toBeVisible();
  });

  test('should login successfully and redirect to dashboard', async ({ page }) => {
    // This test assumes a test user exists
    // In a real scenario, you'd create a test user first
    await page.goto('/login');
    
    // Fill with test credentials
    await page.fill('input[name="email"]', 'test@test.com');
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.locator('button[type="submit"]').click();
    
    // Should redirect to dashboard
    await expect(page).toHaveURL(/.*dashboard/, { timeout: 15000 });
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('should logout successfully', async ({ page }) => {
    // First login
    await page.goto('/login');
    await page.fill('input[name="email"]', 'test@test.com');
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.locator('button[type="submit"]').click();
    
    // Wait for dashboard
    await expect(page).toHaveURL(/.*dashboard/, { timeout: 15000 });
    
    // Find and click logout button
    await page.click('[data-testid="logout-button"]');
    
    // Should redirect to login
    await expect(page).toHaveURL(/.*login/);
    
    // Verify localStorage is cleared
    const authData = await page.evaluate(() => localStorage.getItem('supabase.auth.token'));
    expect(authData).toBeNull();
  });

  test('should protect authenticated routes', async ({ page }) => {
    // Try to access dashboard without authentication
    await page.goto('/dashboard');
    
    // Should redirect to login
    await expect(page).toHaveURL(/.*login/);
  });

  test('should handle forgot password flow', async ({ page }) => {
    await page.goto('/login');
    
    // Click forgot password link
    await page.click('text=Forgot password?');
    
    // Should be on forgot password page
    await expect(page).toHaveURL(/.*forgot-password/);
    await expect(page.locator('input[name="email"]')).toBeVisible();
    
    // Fill email and submit
    await page.fill('input[name="email"]', 'test@test.com');
    await page.locator('button[type="submit"]').click();
    
    // Should show success message
    await expect(page.locator('text=Password reset email sent')).toBeVisible();
  });
});
