import { Router } from 'express';
import { logger } from '../config/logger';
import { WebSocket } from 'ws';
import { supabase } from '../config/supabase';

const router = Router();

// WebSocket client interface
interface WebSocketClient {
  id: string;
  userId: string;
  ws: WebSocket;
  subscribedChannels: Set<string>;
  lastActivity: number;
  messageCount: number;
  isAlive: boolean;
}

// Extend Router to support WebSocket upgrades
declare module 'express-serve-static-core' {
  interface Router {
    ws(path: string, ...handlers: ((ws: WebSocket, req: import('express').Request) => void)[]): void;
  }
}

// WebSocket upgrade handler for /api/v2/dashboard/ws
router.ws('/api/v2/dashboard/ws', async (ws: WebSocket, req: import('express').Request) => {
  // Extract token from query params or headers
  const token = req.query.token as string || req.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    logger.warn('WebSocket connection rejected: No authentication token provided');
    ws.close(1008, 'Unauthorized');
    return;
  }

  try {
    // Verify the token with Supabase
    const { data: { user }, error } = await supabase.auth.getUser(token);
    if (error || !user) {
      logger.warn('WebSocket connection rejected: Invalid token', { error: error?.message });
      ws.close(1008, 'Unauthorized');
      return;
    }

    const userId = user.id;

    logger.info(`WebSocket connection established for user: ${userId}`);

    // Create a client object compatible with WebSocketManager
    const client: WebSocketClient = {
      id: `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId,
      ws,
      subscribedChannels: new Set(),
      lastActivity: Date.now(),
      messageCount: 0,
      isAlive: true
    };

    // For now, we'll manage clients locally since WebSocketManager doesn't expose the clients property
    // In a real implementation, we'd add proper methods to WebSocketManager
    const clients = new Map<string, Set<WebSocketClient>>();
    const userClients = clients.get(userId) || new Set();
    userClients.add(client);
    clients.set(userId, userClients);

    // Send connection confirmation
    ws.send(JSON.stringify({
      type: 'connected',
      userId,
      timestamp: Date.now(),
      message: 'WebSocket connection established'
    }));

    // Handle incoming messages
    ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        logger.debug(`WebSocket message from user ${userId}:`, message);

        // Update activity
        client.lastActivity = Date.now();
        client.messageCount++;

        // Handle different message types
        switch (message.type) {
          case 'ping':
            ws.send(JSON.stringify({
              type: 'pong',
              timestamp: Date.now()
            }));
            break;
          case 'subscribe':
            // Handle subscription to specific channels
            if (message.channel && client.subscribedChannels.size < 20) {
              client.subscribedChannels.add(message.channel);
              ws.send(JSON.stringify({
                type: 'subscription_confirmed',
                channel: message.channel,
                timestamp: Date.now()
              }));
            }
            break;
          case 'unsubscribe':
            // Handle unsubscription from channels
            if (message.channel) {
              client.subscribedChannels.delete(message.channel);
              ws.send(JSON.stringify({
                type: 'unsubscription_confirmed',
                channel: message.channel,
                timestamp: Date.now()
              }));
            }
            break;
          default:
            logger.warn(`Unknown WebSocket message type: ${message.type}`);
        }
      } catch (error) {
        logger.error(`Error processing WebSocket message:`, error);
        ws.send(JSON.stringify({
          type: 'error',
          error: 'Invalid message format',
          timestamp: Date.now()
        }));
      }
    });

    // Handle connection close
    ws.on('close', (code: number, reason: string) => {
      logger.info(`WebSocket connection closed for user ${userId}: ${code} - ${reason}`);
      
      // Remove client from local manager
      const userClients = clients.get(userId);
      if (userClients) {
        userClients.delete(client);
        if (userClients.size === 0) {
          clients.delete(userId);
        }
      }
    });

    // Handle connection errors
    ws.on('error', (error: Error) => {
      logger.error(`WebSocket error for user ${userId}:`, error);
      
      // Remove client from local manager
      const userClients = clients.get(userId);
      if (userClients) {
        userClients.delete(client);
        if (userClients.size === 0) {
          clients.delete(userId);
        }
      }
    });

  } catch (error) {
    logger.error(`Error setting up WebSocket connection:`, error);
    ws.close(1011, 'Internal server error');
  }
});

export { router as websocketRouter };
