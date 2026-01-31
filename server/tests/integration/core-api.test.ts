import request from 'supertest';
import { config } from 'dotenv';

// Load environment variables
config();

// Test configuration
const TEST_PORT = process.env.TEST_PORT || 3003;
const TEST_BASE_URL = `http://localhost:${TEST_PORT}`;

// Mock Express app for testing
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';

describe('Core API Integration Tests', () => {
  let app: express.Application;
  let server: any;

  beforeAll(async () => {
    // Create a minimal Express app for testing
    app = express();
    
    // Apply basic middleware
    app.use(helmet());
    app.use(cors());
    app.use(express.json());
    
    // Health check endpoint
    app.get('/health', (req, res) => {
      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: '2.0.0-test'
      });
    });
    
    // Test API endpoints
    app.get('/api/v2/test', (req, res) => {
      res.json({ success: true, message: 'Test endpoint working' });
    });
    
    // Protected endpoint
    app.get('/api/v2/protected', (req, res) => {
      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ success: false, error: 'Authentication required' });
      }
      res.json({ success: true, message: 'Protected endpoint working' });
    });
    
    // Error handling endpoint
    app.get('/api/v2/error', (req, res) => {
      throw new Error('Test error');
    });
    
    // Error handler
    app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
      res.status(500).json({ success: false, error: err.message });
    });
    
    // 404 handler
    app.use((req, res) => {
      res.status(404).json({ success: false, error: 'Not found' });
    });
    
    // Start test server
    await new Promise<void>((resolve) => {
      server = app.listen(TEST_PORT, () => {
        console.log(`Test server running on port ${TEST_PORT}`);
        resolve();
      });
    });
  });

  afterAll((done) => {
    if (server) {
      server.close(() => {
        console.log('Test server closed');
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

      expect(response.body).toHaveProperty('status', 'ok');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('uptime');
      expect(response.body).toHaveProperty('version', '2.0.0-test');
    });
  });

  describe('Basic API Endpoints', () => {
    it('should handle basic GET request', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/test')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('message', 'Test endpoint working');
    });

    it('should reject unauthenticated requests to protected endpoints', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/protected')
        .expect(401);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body).toHaveProperty('error', 'Authentication required');
    });

    it('should allow authenticated requests to protected endpoints', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/protected')
        .set('Authorization', 'Bearer test-token')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('message', 'Protected endpoint working');
    });
  });

  describe('Error Handling', () => {
    it('should handle server errors gracefully', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/error')
        .expect(500);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body).toHaveProperty('error', 'Test error');
    });

    it('should handle 404 for unknown routes', async () => {
      const response = await request(TEST_BASE_URL)
        .get('/api/v2/unknown-route')
        .expect(404);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body).toHaveProperty('error', 'Not found');
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

  describe('Request Validation', () => {
    it('should handle JSON parsing errors', async () => {
      const response = await request(TEST_BASE_URL)
        .post('/api/v2/test')
        .set('Content-Type', 'application/json')
        .send('invalid-json')
        .expect(400);

      // Express should handle JSON parsing errors automatically
    });

    it('should handle large payloads', async () => {
      const largePayload = 'x'.repeat(10000);
      
      const response = await request(TEST_BASE_URL)
        .post('/api/v2/test')
        .send({ data: largePayload })
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });

  describe('CORS Handling', () => {
    it('should handle CORS preflight requests', async () => {
      const response = await request(TEST_BASE_URL)
        .options('/api/v2/test')
        .expect(204);

      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });
});
