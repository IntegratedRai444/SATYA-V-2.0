import request from 'supertest';
import { createServer } from 'http';
import { config } from 'dotenv';
import { sign } from 'jsonwebtoken';
import { logger } from '../../config/logger';

// Load environment variables
config();

// Test configuration
const TEST_PORT = process.env.TEST_PORT || 3002;
const TEST_BASE_URL = `http://localhost:${TEST_PORT}`;

// Test user data
const TEST_USER = {
  id: 'test-user-123',
  email: 'test@example.com',
  role: 'user',
  email_verified: true
};

// Generate a test JWT token
const generateTestToken = (payload: any = {}) => {
  const jwtSecret = process.env.JWT_SECRET || 'test-secret';
  return sign(
    { ...TEST_USER, ...payload },
    jwtSecret,
    { expiresIn: '1h' }
  );
};

describe('API Integration Tests', () => {
  let server: ReturnType<typeof createServer>;
  let app: any;

  beforeAll(async () => {
    // Set test environment
    process.env.NODE_ENV = 'test';
    process.env.PORT = TEST_PORT.toString();
    
    // Import the Express app dynamically
    const serverModule = await import('../../index');
    
    // Create a test app instance
    const express = require('express');
    app = express();
    
    // Start the test server
    await new Promise<void>((resolve) => {
      server = createServer(app);
      server.listen(TEST_PORT, () => {
        logger.info(`Test server running on port ${TEST_PORT}`);
        resolve();
      });
    });
  });

  afterAll((done) => {
    if (server) {
      server.close(() => {
        logger.info('Test server closed');
        done();
      });
    } else {
      done();
    }
  });

  describe('Health Check', () => {
    it('should return health status', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/health')
        .expect(200);

      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('uptime');
    });
  });

  describe('Authentication', () => {
    it('should reject requests without authentication', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/history')
        .expect(401);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body).toHaveProperty('error');
    });

    it('should accept requests with valid JWT token', async () => {
      const token = generateTestToken();
      
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/history')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });
  });

  describe('API Endpoints', () => {
    const token = generateTestToken();

    it('should get analysis history', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/history')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body.data).toHaveProperty('jobs');
      expect(Array.isArray(response.body.data.jobs)).toBe(true);
    });

    it('should get models information', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/models')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });

    it('should handle file upload validation', async () => {
      const response = await request(TEST_BASE_URL)
        .post('/api/v2/analysis/image')
        .set('Authorization', `Bearer ${token}`)
        .attach('image', Buffer.from('fake-image-data'), 'test.jpg')
        .expect(400); // Should fail due to magic number validation

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('Error Handling', () => {
    const token = generateTestToken();

    it('should handle 404 for unknown routes', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/unknown-route')
        .set('Authorization', `Bearer ${token}`)
        .expect(404);

      expect(response.body).toHaveProperty('success', false);
    });

    it('should handle invalid JSON', async () => {
      const response = await request(TEST_BASE_URL)
        .post('/api/v2/test')
        .set('Content-Type', 'application/json')
        .send('invalid-json')
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('Rate Limiting', () => {
    it('should allow normal request rates', async () => {
      const token = generateTestToken();
      
      // Make a few requests - should all succeed
      for (let i = 0; i < 5; i++) {
        await request(TEST_BASE_URL)
          .get('/api/v2/models')
          .expect(200);
      }
    });
  });

  describe('Security Headers', () => {
    it('should include security headers', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/health')
        .expect(200);

      expect(response.headers).toHaveProperty('x-content-type-options');
      expect(response.headers).toHaveProperty('x-frame-options');
      expect(response.headers).toHaveProperty('x-xss-protection');
    });
  });
});
