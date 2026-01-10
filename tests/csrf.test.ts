import request from 'supertest';
import { app } from '../server/src/app';
import { Server } from 'http';
import { supabase } from '../server/src/config/supabase';

let server: Server;
let csrfToken: string;
let csrfCookie: string[];

beforeAll(async () => {
  // Start the server
  server = app.listen(5001);
  
  // Get a CSRF token for testing
  const response = await request(server)
    .get('/auth/csrf-token')
    .expect(200);
    
  csrfToken = response.body.token;
  csrfCookie = response.headers['set-cookie'];
});

afterAll((done) => {
  // Close the server
  server.close(done);
});

describe('CSRF Protection', () => {
  test('should return 403 when no CSRF token is provided', async () => {
    await request(server)
      .post('/auth/login')
      .send({
        email: 'test@example.com',
        password: 'password123'
      })
      .expect(403)
      .expect('Content-Type', /json/)
      .then(response => {
        expect(response.body).toHaveProperty('success', false);
        expect(response.body).toHaveProperty('code', 'CSRF_TOKEN_REQUIRED');
      });
  });

  test('should return 403 when invalid CSRF token is provided', async () => {
    await request(server)
      .post('/auth/login')
      .set('X-CSRF-Token', 'invalid-token')
      .set('Cookie', csrfCookie)
      .send({
        email: 'test@example.com',
        password: 'password123'
      })
      .expect(403)
      .expect('Content-Type', /json/)
      .then(response => {
        expect(response.body).toHaveProperty('success', false);
        expect(response.body).toHaveProperty('code', 'INVALID_CSRF_TOKEN');
      });
  });

  test('should accept valid CSRF token', async () => {
    // This will fail with 401 (unauthorized) because we're not providing valid credentials
    // but it should pass the CSRF check
    await request(server)
      .post('/auth/login')
      .set('X-CSRF-Token', csrfToken)
      .set('Cookie', csrfCookie)
      .send({
        email: 'test@example.com',
        password: 'wrongpassword'
      })
      .expect(401) // Unauthorized, but CSRF check passed
      .expect('Content-Type', /json/);
  });

  test('should not require CSRF token for GET requests', async () => {
    await request(server)
      .get('/auth/providers')
      .expect(200);
  });
});
