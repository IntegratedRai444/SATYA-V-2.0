/**
 * Basic Health Check Test
 * Tests the most fundamental functionality
 */

import request from 'supertest';
import { describe, it, expect } from '@jest/globals';

// Import the actual app
import { app } from '../index';

describe('Health Check Tests', () => {
  it('should respond to health check', async () => {
    const response = await request(app)
      .get('/api/v2/health')
      .expect(200);

    expect(response.body).toHaveProperty('status');
    expect(response.body).toHaveProperty('timestamp');
  });
});
