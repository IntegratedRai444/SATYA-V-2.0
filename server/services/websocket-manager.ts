import { WebSocketServer, WebSocket, RawData } from 'ws';
import { Server, IncomingMessage } from 'http';
import { URLSearchParams } from 'url';
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
  SUBSCRIBE_JOB: 'SUBSCRIBE_JOB',
  UNSUBSCRIBE_JOB: 'UNSUBSCRIBE_JOB',
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
  type: z.enum(['subscribe', 'unsubscribe', 'message', 'ping', 'pong', 'SUBSCRIBE_JOB', 'UNSUBSCRIBE_JOB']),
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

  // Extract token from query parameter or Authorization header
  private extractToken(info: { req: IncomingMessage; origin: string; secure: boolean }): string | null {
    // Debug logging to see what we're receiving
    logger.debug('WebSocket token extraction attempt', {
      url: info.req.url,
      headers: Object.keys(info.req.headers),
      hasAuthHeader: !!info.req.headers['authorization']
    });
    
    // Check Authorization header first
    const authHeader = info.req.headers['authorization'];
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const token = authHeader.substring(7); // Remove 'Bearer ' prefix
      logger.debug('Token extracted from Authorization header');
      return token;
    }
    
    // Check query parameter - the URL might be just the path, so we need to handle it differently
    const requestUrl = info.req.url;
    logger.debug('Raw request URL:', requestUrl);
    
    if (requestUrl) {
      // Method 1: Check if URL contains token directly
      if (requestUrl.includes('?token=')) {
        const urlParts = requestUrl.split('?');
        if (urlParts.length > 1) {
          const queryParams = new URLSearchParams(urlParts[1]);
          const token = queryParams.get('token');
          if (token) {
            logger.debug('Token extracted from query parameter (method 1)');
            return token;
          }
        }
      }
      
      // Method 2: Check if the original request URL (if available) has the token
      // Sometimes the full URL is available in headers or as a property
      const fullUrl = (info.req as unknown as { originalUrl?: string; url?: string }).originalUrl || 
                      (info.req as unknown as { originalUrl?: string; url?: string }).url;
      if (fullUrl && fullUrl.includes('?token=')) {
        const urlParts = fullUrl.split('?');
        if (urlParts.length > 1) {
          const queryParams = new URLSearchParams(urlParts[1]);
          const token = queryParams.get('token');
          if (token) {
            logger.debug('Token extracted from full URL (method 2)');
            return token;
          }
        }
      }
    }
    
    logger.warn('No token found in WebSocket request', {
      url: requestUrl,
      availableHeaders: Object.keys(info.req.headers)
    });
    
    return null;
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
      path: '/api/v2/dashboard/ws', // Match frontend connection URL
      verifyClient: async (info: { req: IncomingMessage; origin: string; secure: boolean; }, callback: (res: boolean, code?: number, message?: string) => void) => {
        logger.debug('WebSocket connection attempt', {
          url: info.req.url,
          headers: info.req.headers,
          host: info.req.headers.host
        });

        // Extract token from query parameter or Authorization header
        const token = this.extractToken(info);
        
        if (!token) {
          logger.warn('WebSocket connection rejected: No token provided');
          callback(false, 1008, 'No authentication token provided');
          return;
        }

        // Verify JWT token
        try {
          const { supabase } = await import('../config/supabase');
          const { data: { user } } = await supabase.auth.getUser(token);
          
          if (!user) {
            logger.warn('WebSocket connection rejected: Invalid token');
            callback(false, 1008, 'Invalid authentication token');
            return;
          }

          logger.info('WebSocket connection authenticated successfully', { userId: user.id, email: user.email });
          callback(true);
        } catch (error) {
          logger.error('WebSocket authentication error:', error);
          callback(false, 1008, 'Authentication failed');
        }
      }
    });

    this.wss.on('connection', (ws: WebSocketClient, req: IncomingMessage) => {
      this.handleConnection(ws, req);
    });

    logger.info('[WS SERVER] WebSocket server initialized');
  }

  private async handleConnection(ws: WebSocketClient, req: IncomingMessage): Promise<void> {
    // Token was already verified in verifyClient, so we can extract user info from the verified connection
    // The verifyClient callback already authenticated the user, so we can proceed with connection setup
    
    // Extract token for user info (we know it's valid since verifyClient passed)
    const token = this.extractToken({ req, origin: req.headers.origin || '', secure: false });
    
    if (!token) {
      // This shouldn't happen if verifyClient worked, but add safety check
      logger.warn('WebSocket connection failed: No token in handleConnection');
      ws.terminate();
      return;
    }

    // Get user info from token (this should work since verifyClient already validated it)
    const authResult = await supabaseAuthService.verifyToken(token);
    if (!authResult) {
      logger.warn('WebSocket connection rejected: Could not verify token in handleConnection');
      ws.terminate();
      return;
    }

    logger.info('WebSocket connection authenticated successfully', { 
      userId: authResult.userId, 
      email: authResult.email 
    });
    
    // Initialize client properties
    ws.clientId = this.generateClientId();
    ws.userId = authResult.userId;
    ws.username = authResult.username;
    ws.sessionId = authResult.sessionId;
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
    if (!this.clients.has(ws.userId)) {
      this.clients.set(ws.userId, new Set());
    }
    this.clients.get(ws.userId)!.add(ws);

    logger.info('[WS CONNECT] New WebSocket connection', {
      clientId: ws.clientId,
      userId: ws.userId,
      username: ws.username,
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

      case 'SUBSCRIBE_JOB':
        if (message.payload?.jobId) {
          const channel = `job:${message.payload.jobId}`;
          ws.subscribedChannels.add(channel);
          this.sendToClient(ws, {
            type: 'subscription_confirmed',
            channel,
            timestamp: Date.now(),
            payload: { jobId: message.payload.jobId }
          });
        }
        break;

      case 'UNSUBSCRIBE_JOB':
        if (message.payload?.jobId) {
          const channel = `job:${message.payload.jobId}`;
          ws.subscribedChannels.delete(channel);
          this.sendToClient(ws, {
            type: 'unsubscription_confirmed',
            channel,
            timestamp: Date.now(),
            payload: { jobId: message.payload.jobId }
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

  // Send job update to specific user
  public sendJobUpdate(userId: string, jobId: string, status: string, progress?: number, message?: string): void {
    const jobChannel = `job:${jobId}`;
    const userClients = this.clients.get(userId);
    
    if (userClients) {
      userClients.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN && ws.subscribedChannels.has(jobChannel)) {
          this.sendToClient(ws, {
            type: 'JOB_PROGRESS',
            jobId,
            payload: {
              jobId,
              status,
              progress: progress || 0,
              message: message || '',
              timestamp: new Date().toISOString()
            },
            timestamp: Date.now()
          });
        }
      });
    }
  }

  // Send job completion to specific user
  public sendJobCompleted(userId: string, jobId: string, result?: unknown): void {
    const jobChannel = `job:${jobId}`;
    const userClients = this.clients.get(userId);
    
    if (userClients) {
      userClients.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN && ws.subscribedChannels.has(jobChannel)) {
          this.sendToClient(ws, {
            type: 'JOB_COMPLETED',
            jobId,
            payload: {
              jobId,
              status: 'completed',
              result,
              timestamp: new Date().toISOString()
            },
            timestamp: Date.now()
          });
        }
      });
    }
  }

  // Send job failure to specific user
  public sendJobFailed(userId: string, jobId: string, error: string): void {
    const jobChannel = `job:${jobId}`;
    const userClients = this.clients.get(userId);
    
    if (userClients) {
      userClients.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN && ws.subscribedChannels.has(jobChannel)) {
          this.sendToClient(ws, {
            type: 'JOB_FAILED',
            jobId,
            payload: {
              jobId,
              status: 'failed',
              error,
              timestamp: new Date().toISOString()
            },
            timestamp: Date.now()
          });
        }
      });
    }
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
