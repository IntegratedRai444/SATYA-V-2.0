import { WebSocket } from 'ws';
import { logger } from '../../config/logger';
import { config } from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { createServer } from 'http';
import request from 'supertest';
import { sign } from 'jsonwebtoken';
import { JWT_SECRET } from '../../config/constants';

// Import the server app - we'll create a test instance
import('../../index').then((module) => {
  // We'll handle this in the setup
});

// Load environment variables
config();

// Test user data
const TEST_USER = {
  userId: 'test-user-123',
  sessionId: 'test-session-456',
};

// Generate a test JWT token
const generateTestToken = (payload: any = {}) => {
  return sign(
    { ...TEST_USER, ...payload },
    JWT_SECRET,
    { expiresIn: '1h' }
  );
};

describe('WebSocket Error Scenarios', () => {
  let server: ReturnType<typeof createServer>;
  let ws: WebSocket;
  const TEST_PORT = process.env.TEST_PORT || 3001;
  const TEST_URL = `ws://localhost:${TEST_PORT}/ws`;
  
  // Set a default test timeout
  jest.setTimeout(10000);

  beforeAll((done) => {
    server = createServer(app);
    server.listen(TEST_PORT, () => {
      logger.info(`Test server running on port ${TEST_PORT}`);
      done();
    });
  });

  afterAll((done) => {
    // Close any open WebSocket connections
    if (ws) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else {
        ws.terminate();
      }
    }
    
    // Close the server
    server.close(() => {
      logger.info('Test server closed');
      done();
    });
  });

  afterEach((done) => {
    // Ensure WebSocket is properly closed after each test
    if (ws) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.once('close', () => done());
        ws.close();
      } else {
        ws.terminate();
        done();
      }
    } else {
      done();
    }
  });

  it('should handle invalid message format', (done) => {
    // Mock JWT token for testing
    const testToken = 'test-token';
    
    // Setup WebSocket connection with token in query parameter
    ws = new WebSocket(`${TEST_URL}?token=${testToken}`);

    // Set a timeout for the test
    const testTimeout = setTimeout(() => {
      if (ws.readyState === WebSocket.OPEN) ws.close();
      done(new Error('Test timeout'));
    }, 5000);

    ws.on('open', () => {
      try {
        // Send invalid JSON
        ws.send('invalid-json');
      } catch (error) {
        clearTimeout(testTimeout);
        done(error);
      }
    });

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        if (message.type === 'error') {
          clearTimeout(testTimeout);
          expect(message.error).toContain('Invalid message format');
          ws.close();
          done();
        }
      } catch (error) {
        clearTimeout(testTimeout);
        ws.close();
        done(error);
      }
    });
  });

  it('should handle unauthorized connections', (done) => {
    // Test without token
    ws = new WebSocket(TEST_URL);

    ws.on('error', (error: Error) => {
      expect(error.message).toContain('401');
      done();
    });
  });

  it('should handle invalid JWT token', (done) => {
    ws = new WebSocket(`${TEST_URL}?token=invalid-token`);

    ws.on('error', (error: Error) => {
      expect(error.message).toContain('401');
      done();
    });
  });

  it('should handle malformed messages', (done) => {
    const testToken = generateTestToken();
    ws = new WebSocket(`${TEST_URL}?token=${testToken}`);

    ws.on('open', () => {
      // Send malformed message (not a string)
      ws.send(123 as any);
    });

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        if (message.type === 'error') {
          expect(message.error).toBeDefined();
          done();
        }
      } catch (error) {
        done(error);
      }
    });
  });

  it('should handle rate limiting', (done) => {
    const testToken = generateTestToken();
    ws = new WebSocket(`${TEST_URL}?token=${testToken}`);
    let messageCount = 0;
    const maxMessages = 10; // Adjust based on your rate limiter configuration

    ws.on('open', () => {
      // Send multiple messages quickly to trigger rate limiting
      const sendMessages = () => {
        for (let i = 0; i < maxMessages; i++) {
          ws.send(JSON.stringify({ type: 'ping', data: { count: i } }));
        }
      };
      sendMessages();
    });

    ws.on('message', (data) => {
      messageCount++;
      const message = JSON.parse(data.toString());
      
      if (message.type === 'error' && message.error?.includes('rate limit')) {
        expect(messageCount).toBeGreaterThan(0);
        done();
      }
    });
  });

  it('should handle connection drops and reconnections', (done) => {
    ws = new WebSocket(TEST_URL, {
      headers: { 'Authorization': 'Bearer test-token' }
    });

    let reconnected = false;
    let initialConnectionId: string;

    ws.on('open', () => {
      // Get initial connection ID
      ws.send(JSON.stringify({ type: 'get_connection_id' }));
    });

    ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      
      if (message.type === 'connection_id') {
        if (!initialConnectionId) {
          initialConnectionId = message.connectionId;
          // Simulate connection drop
          ws.terminate();
          
          // Reconnect after a short delay
          setTimeout(() => {
            ws = new WebSocket(TEST_URL, {
              headers: { 'Authorization': 'Bearer test-token' }
            });
            
            ws.on('open', () => {
              ws.send(JSON.stringify({ type: 'get_connection_id' }));
            });
            
            ws.on('message', (data) => {
              const msg = JSON.parse(data.toString());
              if (msg.type === 'connection_id') {
                expect(msg.connectionId).not.toBe(initialConnectionId);
                reconnected = true;
                done();
              }
            });
          }, 1000);
        }
      }
    });
  });

  it('should handle rate limiting', (done) => {
    // Create multiple connections to test rate limiting
    const connections: WebSocket[] = [];
    let rateLimited = false;
    
    // Create more connections than allowed by rate limit
    for (let i = 0; i < 10; i++) {
      const ws = new WebSocket(TEST_URL, {
        headers: { 'Authorization': `Bearer test-token-${i}` }
      });
      
      ws.on('error', (error) => {
        if (error.message.includes('429')) {
          rateLimited = true;
          // Clean up
          connections.forEach(conn => conn.terminate());
          expect(rateLimited).toBe(true);
          done();
        }
      });
      
      connections.push(ws);
    }
  });

  it('should handle malformed messages', (done) => {
    ws = new WebSocket(TEST_URL, {
      headers: { 'Authorization': 'Bearer test-token' }
    });

    ws.on('open', () => {
      // Send message with missing required fields
      ws.send(JSON.stringify({ type: 'invalid_type' }));
    });

    ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      if (message.type === 'error') {
        expect(message.error).toBeDefined();
        done();
      }
    });
  });
});
