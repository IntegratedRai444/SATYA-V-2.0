import { test, expect } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

test.describe('Deepfake Analysis Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto('/login');
    await page.fill('input[name="email"]', 'test@test.com');
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.locator('button[type="submit"]').click();
    await expect(page).toHaveURL(/.*dashboard/, { timeout: 15000 });
  });

  test('should navigate to image analysis page', async ({ page }) => {
    await page.click('[data-testid="nav-analysis"]');
    await page.click('[data-testid="image-analysis"]');
    
    await expect(page).toHaveURL(/.*analysis\/image/);
    await expect(page.locator('h1')).toContainText('Image Analysis');
  });

  test('should upload image file successfully', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Create a test image file (you'd use a real test image in practice)
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    
    // Upload file
    await page.setInputFiles('input[type="file"]', testImagePath);
    
    // Check file preview appears
    await expect(page.locator('[data-testid="file-preview"]')).toBeVisible();
    await expect(page.locator('[data-testid="analyze-button"]')).toBeEnabled();
  });

  test('should show file size validation', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Try to upload a file that's too large (mock this by checking validation)
    await page.setInputFiles('input[type="file"]', 'large-file.jpg');
    
    // Should show size error
    await expect(page.locator('text=File size too large')).toBeVisible();
  });

  test('should start image analysis', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Upload test image
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    
    // Start analysis
    await page.click('[data-testid="analyze-button"]');
    
    // Should show progress indicator
    await expect(page.locator('[data-testid="analysis-progress"]')).toBeVisible();
    await expect(page.locator('text=Analyzing')).toBeVisible();
  });

  test('should display analysis results', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Upload and analyze
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    await page.click('[data-testid="analyze-button"]');
    
    // Wait for results (mock this in tests)
    await expect(page.locator('[data-testid="analysis-results"]')).toBeVisible({ timeout: 30000 });
    
    // Check result elements
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    await expect(page.locator('[data-testid="analysis-verdict"]')).toBeVisible();
    await expect(page.locator('[data-testid="cryptographic-proof"]')).toBeVisible();
  });

  test('should handle video analysis', async ({ page }) => {
    await page.goto('/analysis/video');
    
    // Upload test video
    const testVideoPath = path.join(__dirname, 'fixtures', 'test-video.mp4');
    await page.setInputFiles('input[type="file"]', testVideoPath);
    
    // Start analysis
    await page.click('[data-testid="analyze-button"]');
    
    // Video analysis takes longer, check for progress
    await expect(page.locator('[data-testid="analysis-progress"]')).toBeVisible();
    await expect(page.locator('text=Processing video')).toBeVisible();
  });

  test('should handle audio analysis', async ({ page }) => {
    await page.goto('/analysis/audio');
    
    // Upload test audio
    const testAudioPath = path.join(__dirname, 'fixtures', 'test-audio.wav');
    await page.setInputFiles('input[type="file"]', testAudioPath);
    
    // Start analysis
    await page.click('[data-testid="analyze-button"]');
    
    // Check for audio-specific progress
    await expect(page.locator('[data-testid="analysis-progress"]')).toBeVisible();
    await expect(page.locator('text=Analyzing audio')).toBeVisible();
  });

  test('should handle multimodal analysis', async ({ page }) => {
    await page.goto('/analysis/multimodal');
    
    // Upload multiple files
    await page.setInputFiles('input[type="file"]', [
      path.join(__dirname, 'fixtures', 'test-image.jpg'),
      path.join(__dirname, 'fixtures', 'test-audio.wav')
    ]);
    
    // Start analysis
    await page.click('[data-testid="analyze-button"]');
    
    // Check for multimodal progress
    await expect(page.locator('[data-testid="analysis-progress"]')).toBeVisible();
    await expect(page.locator('text=Multimodal analysis')).toBeVisible();
  });

  test('should show error for unsupported file type', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Try to upload unsupported file
    await page.setInputFiles('input[type="file"]', 'test-file.txt');
    
    // Should show error
    await expect(page.locator('text=Unsupported file type')).toBeVisible();
  });

  test('should allow downloading analysis report', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Complete an analysis first
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    await page.click('[data-testid="analyze-button"]');
    
    // Wait for results
    await expect(page.locator('[data-testid="analysis-results"]')).toBeVisible({ timeout: 30000 });
    
    // Download report
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="download-report"]');
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toMatch(/analysis-report.*\.pdf/);
  });

  test('should save analysis to history', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Complete analysis
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    await page.click('[data-testid="analyze-button"]');
    
    // Wait for results
    await expect(page.locator('[data-testid="analysis-results"]')).toBeVisible({ timeout: 30000 });
    
    // Save to history
    await page.click('[data-testid="save-to-history"]');
    
    // Check success message
    await expect(page.locator('text=Analysis saved to history')).toBeVisible();
    
    // Navigate to history to verify
    await page.goto('/history');
    await expect(page.locator('[data-testid="history-item"]')).toBeVisible();
  });

  test('should handle analysis cancellation', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Start analysis
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    await page.click('[data-testid="analyze-button"]');
    
    // Cancel analysis
    await page.click('[data-testid="cancel-analysis"]');
    
    // Should show cancellation message
    await expect(page.locator('text=Analysis cancelled')).toBeVisible();
    await expect(page.locator('[data-testid="analysis-progress"]')).not.toBeVisible();
  });

  test('should show real-time progress updates', async ({ page }) => {
    await page.goto('/analysis/image');
    
    // Start analysis
    const testImagePath = path.join(__dirname, 'fixtures', 'test-image.jpg');
    await page.setInputFiles('input[type="file"]', testImagePath);
    await page.click('[data-testid="analyze-button"]');
    
    // Check for WebSocket progress updates
    await expect(page.locator('[data-testid="progress-percentage"]')).toBeVisible();
    
    // Progress should update (mock this in tests)
    const progressElement = page.locator('[data-testid="progress-percentage"]');
    await expect(progressElement).toHaveText(/\d+%/);
  });
});
