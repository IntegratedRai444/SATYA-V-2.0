import { WebSocketServer, WebSocket, RawData } from 'ws';
import { Server, IncomingMessage } from 'http';
import { logger } from '../config/logger';
import { z } from 'zod';
import { supabase } from '../config/supabase';

// Timer imports
import { setInterval, clearInterval } from 'timers';

// Timer types for compatibility
type Timer = ReturnType<typeof setInterval> & { ref?: () => void; unref?: () => void; };

// Message size limits
const MESSAGE_LIMITS = {
  MAX_MESSAGE_SIZE: 1024 * 1024, // 1MB max message size
  MAX_CONNECTIONS_PER_IP: 5, // Max concurrent connections per IP
  MESSAGES_PER_MINUTE: 100, // Max messages per minute per connection
  MESSAGE_RATE_WINDOW_MS: 60 * 1000, // 1 minute window
  BLOCK_DURATION_MS: 5 * 60 * 1000, // 5 minute block duration
  PING_INTERVAL: 30000, // 30 seconds
  PONG_TIMEOUT: 10000, // 10 seconds
  MAX_PAYLOAD_SIZE: 10 * 1024 * 1024, // 10MB max payload size
  MAX_QUEUE_SIZE: 100, // Max queued messages per client
  MAX_SUBSCRIPTIONS: 20, // Max subscriptions per connection
  MAX_CHANNEL_LENGTH: 100, // Max channel name length
  MAX_HEADER_SIZE: 4096, // 4KB max header size
  MAX_WEBSOCKET_FRAME: 16 * 1024, // 16KB max WebSocket frame size
} as const;

// WebSocket message type
const WebSocketMessageType = {
  MESSAGE: 'message',
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  PING: 'ping',
  PONG: 'pong',
  ERROR: 'error',
  CONNECTED: 'connected',
  SUBSCRIPTION_CONFIRMED: 'subscription_confirmed',
  JOB_STATUS: 'job_status',
  UNSUBSCRIPTION_CONFIRMED: 'unsubscription_confirmed',
  RECONNECT_REQUIRED: 'reconnect_required',
  JOB_STARTED: 'JOB_STARTED',
  JOB_COMPLETED: 'JOB_COMPLETED',
  JOB_FAILED: 'JOB_FAILED',
  JOB_ERROR: 'JOB_ERROR',
  DASHBOARD_UPDATE: 'dashboard_update',
  JOB_PROGRESS: 'JOB_PROGRESS',
  JOB_STAGE_UPDATE: 'JOB_STAGE_UPDATE',
  JOB_METRICS: 'JOB_METRICS',
  SUBSCRIBE_JOB: 'subscribe_job',
  UNSUBSCRIBE_JOB: 'unsubscribe_job',
  MESSAGE_ACK: 'message_ack'
} as const;

// WebSocket message type definition
type WebSocketMessage = {
  type: typeof WebSocketMessageType[keyof typeof WebSocketMessageType];
  channel?: string;
  payload?: Record<string, unknown>;
  requestId?: string;
  timestamp?: number;
  jobId?: string;
  error?: {
    code: string;
    message: string;
    timestamp?: number;
    requestId?: string;
    details?: unknown;
  };
  data?: unknown;
};

// Supabase Auth Service
const supabaseAuthService = {
  verifyToken: async (token: string) => {
    try {
      const { data: { user }, error } = await supabase.auth.getUser(token);
      if (error || !user) {
        return null;
      }
      return {
        userId: user.id,
        username: user.email || '',
        email: user.email,
        sessionId: user.id // Use user ID as session identifier
      };
    } catch (e) {
      return null;
    }
  }
};

// Define WebSocket message schema
const messageSchema = z.object({
  type: z.enum(['subscribe', 'unsubscribe', 'message', 'ping', 'pong']),
  channel: z.string().optional(),
  payload: z.record(z.string(), z.unknown()).optional(),
  requestId: z.string().uuid().optional(),
  timestamp: z.number().int().positive().optional(),
  jobId: z.string().optional(),
  error: z.object({
    code: z.string(),
    message: z.string(),
    timestamp: z.number().optional()
  }).optional(),
  data: z.unknown().optional()
});

interface WebSocketClient extends WebSocket {
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  reconnectTimeout: number | null;
  isAlive: boolean;
  lastActivity: number;
  messageQueue: Array<{ data: unknown; timestamp: number }>;
  maxQueueSize: number;
  userId: string;
  username?: string;
  sessionId: string;
  clientId: string;
  ipAddress: string;
  connectedAt: number;
  subscribedChannels: Set<string>;
  messageCount: number;
  lastMessageTime: number;
  jobId?: string;
  connectionTime: number;
  [key: string]: unknown; // Index signature for dynamic properties
}

export class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients = new Map<string, Set<WebSocketClient>>();
  private blockedIPs = new Map<string, number>();
  private messageRates = new Map<string, { count: number; resetTime: number }>();
  private pingInterval: Timer | null = null;

  constructor() {
    this.setupPingInterval();
  }

  // Generate a unique client ID
  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Parse incoming WebSocket message
  private parseMessage(data: RawData): WebSocketMessage | null {
    try {
      const message = JSON.parse(data.toString());
      return messageSchema.parse(message);
    } catch (error) {
      logger.error('Failed to parse WebSocket message', {
        error: error instanceof Error ? error.message : 'Unknown error',
        data: data.toString().substring(0, 100) // Log first 100 chars to avoid huge logs
      });
      return null;
    }
  }

  // Add missing method implementations
  private handleDisconnect(ws: WebSocketClient): void {
    logger.info('[WS DISCONNECT] Client disconnected', {
      clientId: ws.clientId,
      userId: ws.userId,
      username: ws.username
    });

    this.removeClient(ws);
  }

  private sendToClient(ws: WebSocketClient, message: WebSocketMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
      } catch (error) {
        logger.error('Error sending message to client', {
          clientId: ws.clientId,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
  }

  private setupPingInterval(): void {
    this.pingInterval = setInterval(() => {
      this.clients.forEach((clientSet) => {
        clientSet.forEach((ws) => {
          if (ws.isAlive === false) {
            ws.terminate();
            return;
          }
          ws.isAlive = false;
          ws.ping();
        });
      });
    }, MESSAGE_LIMITS.PING_INTERVAL);
  }

  // Public methods
  public initialize(server: Server): void {
    this.wss = new WebSocketServer({ 
      server, 
      path: '/api/v2/dashboard/ws',
      verifyClient: async (info: { req: IncomingMessage; origin: string; secure: boolean; }, callback: (res: boolean, code?: number, message?: string) => void) => {
        const token = info.req.headers['authorization']?.replace('Bearer ', '');
        
        if (!token) {
          callback(false, 401, 'Unauthorized');
          return;
        }

        const user = await supabaseAuthService.verifyToken(token);
        if (!user) {
          callback(false, 401, 'Invalid token');
          return;
        }

        callback(true);
      }
    });

    this.wss.on('connection', (ws: WebSocketClient, req: IncomingMessage) => {
      this.handleConnection(ws, req);
    });

    logger.info('[WS SERVER] WebSocket server initialized');
  }

  private async handleConnection(ws: WebSocketClient, req: IncomingMessage): Promise<void> {
    const token = req.headers['authorization']?.replace('Bearer ', '');
    const user = token ? await supabaseAuthService.verifyToken(token) : null;

    if (!user) {
      ws.close(1008, 'Authentication failed');
      return;
    }

    // Initialize client properties
    ws.clientId = this.generateClientId();
    ws.userId = user.userId;
    ws.username = user.username;
    ws.sessionId = user.sessionId;
    ws.ipAddress = req.socket.remoteAddress || 'unknown';
    ws.connectedAt = Date.now();
    ws.connectionTime = Date.now();
    ws.isAlive = true;
    ws.lastActivity = Date.now();
    ws.messageQueue = [];
    ws.maxQueueSize = MESSAGE_LIMITS.MAX_QUEUE_SIZE;
    ws.subscribedChannels = new Set();
    ws.messageCount = 0;
    ws.lastMessageTime = 0;
    ws.reconnectAttempts = 0;
    ws.maxReconnectAttempts = 5;
    ws.reconnectTimeout = null;

    // Add to clients map
    if (!this.clients.has(user.userId)) {
      this.clients.set(user.userId, new Set());
    }
    this.clients.get(user.userId)!.add(ws);

    logger.info('[WS CONNECT] New WebSocket connection', {
      clientId: ws.clientId,
      userId: user.userId,
      username: user.username,
      ipAddress: ws.ipAddress
    });

    // Setup event handlers
    ws.on('pong', () => {
      ws.isAlive = true;
      ws.lastActivity = Date.now();
    });

    ws.on('message', (data: RawData) => {
      this.handleMessage(ws, data);
    });

    ws.on('close', () => {
      this.handleDisconnect(ws);
    });

    ws.on('error', (error) => {
      logger.error('WebSocket error:', error);
      this.handleDisconnect(ws);
    });

    // Send welcome message
    this.sendToClient(ws, {
      type: WebSocketMessageType.CONNECTED,
      timestamp: Date.now(),
      payload: {
        clientId: ws.clientId,
        userId: ws.userId,
        message: 'Connection established successfully'
      }
    });
  }

  private handleMessage(ws: WebSocketClient, data: RawData): void {
    const message = this.parseMessage(data);
    if (!message) return;

    ws.lastActivity = Date.now();
    ws.messageCount++;
    ws.lastMessageTime = Date.now();

    switch (message.type) {
      case 'ping':
        this.sendToClient(ws, {
          type: 'pong',
          timestamp: Date.now()
        });
        break;
      
      case 'subscribe':
        if (message.channel) {
          ws.subscribedChannels.add(message.channel);
          this.sendToClient(ws, {
            type: 'subscription_confirmed',
            channel: message.channel,
            timestamp: Date.now()
          });
        }
        break;
      
      case 'unsubscribe':
        if (message.channel) {
          ws.subscribedChannels.delete(message.channel);
          this.sendToClient(ws, {
            type: 'unsubscription_confirmed',
            channel: message.channel,
            timestamp: Date.now()
          });
        }
        break;
      
      default:
        logger.warn('Unknown message type', { type: message.type });
    }
  }

  // Health monitoring methods
  public getConnectedClientsCount(): number {
    let count = 0;
    for (const clientSet of this.clients.values()) {
      count += clientSet.size;
    }
    return count;
  }

  public isHealthy(): boolean {
    const now = Date.now();
    const fiveMinutesAgo = now - 5 * 60 * 1000;
    let totalConnections = 0;
    let healthyConnections = 0;

    for (const clientSet of this.clients.values()) {
      for (const client of clientSet) {
        totalConnections++;
        if (client.lastActivity > fiveMinutesAgo && client.isAlive) {
          healthyConnections++;
        }
      }
    }

    if (totalConnections === 0) return true;
    const healthRatio = healthyConnections / totalConnections;
    return healthRatio >= 0.95;
  }

  public removeClient(client: WebSocketClient): void {
    const userClients = this.clients.get(client.userId);
    if (userClients) {
      userClients.delete(client);
      if (userClients.size === 0) {
        this.clients.delete(client.userId);
      }
    }
  }

  public getConnectionStats(): {
    totalConnections: number;
    activeConnections: number;
    connectionsByUser: Record<string, number>;
    averageConnectionTime: number;
  } {
    const now = Date.now();
    const stats = {
      totalConnections: 0,
      activeConnections: 0,
      connectionsByUser: {} as Record<string, number>,
      averageConnectionTime: 0
    };

    let totalConnectionTime = 0;
    let connectionCount = 0;

    for (const [userId, clientSet] of this.clients.entries()) {
      const userConnectionCount = clientSet.size;
      stats.connectionsByUser[userId] = userConnectionCount;
      stats.totalConnections += userConnectionCount;

      for (const client of clientSet) {
        if (client.isAlive) {
          stats.activeConnections++;
        }

        const connectionTime = now - client.connectionTime;
        totalConnectionTime += connectionTime;
        connectionCount++;
      }
    }

    if (connectionCount > 0) {
      stats.averageConnectionTime = totalConnectionTime / connectionCount;
    }

    return stats;
  }

  private sendToUser(userId: string, message: WebSocketMessage): void {
    const userClients = this.clients.get(userId);
    if (userClients) {
      let sentCount = 0;
      userClients.forEach((ws) => {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            this.sendToClient(ws, message);
            sentCount++;
          }
        } catch (error) {
          logger.error('Error sending message to client:', {
            clientId: ws.clientId,
            userId,
            messageType: message.type
          });
        }
      });

      logger.debug('Message sent to user clients', {
        userId,
        messageType: message.type,
        sentCount,
        totalClients: userClients.size
      });
    }
  }

  public sendEventToUser(userId: string, message: WebSocketMessage): void {
    this.sendToUser(userId, message);
  }

  public shutdown(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }

    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }

    this.clients.clear();
    logger.info('[WS SERVER] WebSocket server shutdown complete');
  }
}

// Export singleton instance
const webSocketManager = new WebSocketManager();
export default webSocketManager;

// Add global error handlers
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception in WebSocket manager', {
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined
  });
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled promise rejection in WebSocket manager', {
    reason,
    promise
  });
});

// Handle graceful shutdown
const shutdown = (signal: string) => {
  logger.info(`${signal} received, cleaning up WebSocket manager...`);
  webSocketManager.shutdown();
  process.exit(0);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
