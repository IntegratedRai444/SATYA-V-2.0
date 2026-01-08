import { WebSocket } from 'ws';
import { logger } from '../../config/logger';
import { config } from 'dotenv';
import { v4 as uuidv4 } from 'uuid';

config();

describe('WebSocket Load Testing', () => {
  const TEST_URL = `ws://localhost:${process.env.PORT || 3000}/api/v2/ws`;
  const CONCURRENT_CONNECTIONS = 100;
  const MESSAGES_PER_CONNECTION = 50;
  const TEST_DURATION_MS = 60000; // 1 minute

  let successfulConnections = 0;
  let failedConnections = 0;
  let messagesSent = 0;
  let messagesReceived = 0;
  let connections: WebSocket[] = [];

  beforeAll(() => {
    logger.info('Starting WebSocket load test...');
  });

  afterAll(() => {
    // Close all connections
    connections.forEach(ws => ws.terminate());
    logger.info('Load test completed', {
      successfulConnections,
      failedConnections,
      messagesSent,
      messagesReceived,
      successRate: (successfulConnections / (successfulConnections + failedConnections)) * 100 + '%'
    });
  });

  it(`should handle ${CONCURRENT_CONNECTIONS} concurrent connections`, async () => {
    const connectionPromises = Array(CONCURRENT_CONNECTIONS).fill(null).map((_, i) => 
      new Promise<void>((resolve) => {
        const ws = new WebSocket(TEST_URL, {
          headers: { 'Authorization': `Bearer test-token-${i}` }
        });

        ws.on('open', () => {
          successfulConnections++;
          connections.push(ws);
          
          // Send test messages
          for (let j = 0; j < MESSAGES_PER_CONNECTION; j++) {
            const message = {
              type: 'ping',
              requestId: uuidv4(),
              timestamp: Date.now()
            };
            ws.send(JSON.stringify(message));
            messagesSent++;
          }
          
          resolve();
        });

        ws.on('message', (data) => {
          messagesReceived++;
        });

        ws.on('error', (error) => {
          failedConnections++;
          logger.error('Connection error:', error);
          resolve();
        });
      })
    );

    await Promise.all(connectionPromises);
    
    // Keep connections alive for the test duration
    await new Promise(resolve => setTimeout(resolve, TEST_DURATION_MS));
    
    expect(successfulConnections).toBeGreaterThanOrEqual(CONCURRENT_CONNECTIONS * 0.95); // 95% success rate
  });
});
