import { WebSocketServer, WebSocket, RawData } from 'ws';
import { Server, IncomingMessage } from 'http';
import { verify, JwtPayload } from 'jsonwebtoken';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { logger } from '../config/logger';
import { z } from 'zod';
import { EventEmitter } from 'events';
import { JWT_SECRET } from '../config/constants';

// Define JWT Payload interface
interface JwtUserPayload extends JwtPayload {
  userId: string;
  username: string;
  sessionId: string;
  iat?: number;
  exp?: number;
}

// Rate limiting configurations
const rateLimitRules = {
  websocket: {
    windowMs: 60 * 1000, // 1 minute window
    maxConnections: 10, // Max connections per IP
    messageRate: 100, // Max messages per minute per connection
    blockDuration: 5 * 60 * 1000, // 5 minutes block duration
  },
  api: {
    windowMs: 60 * 1000, // 1 minute window
    max: 100, // Max requests per windowMs
  },
};

// Rate limiting configuration
const rateLimiter = new RateLimiterMemory({
  points: rateLimitRules.websocket.messageRate, // messages per minute
  duration: 60, // per 60 seconds per IP
  blockDuration: rateLimitRules.websocket.blockDuration / 1000, // Convert to seconds
});

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
  JOB_STARTED: 'job_started',
  JOB_COMPLETED: 'job_completed',
  DASHBOARD_UPDATE: 'dashboard_update',
  JOB_PROGRESS: 'job_progress',
  JOB_STAGE_UPDATE: 'job_stage_update',
  JOB_METRICS: 'job_metrics',
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
    details?: any;
  };
  data?: unknown;
};

// Helper to extract token from URL query
function extractTokenFromQuery(url: string): string | null {
  try {
    const queryString = url.split('?')[1];
    if (!queryString) return null;
    
    const params = new URLSearchParams(queryString);
    return params.get('token');
  } catch (e) {
    return null;
  }
}

// JWT Auth Service
const jwtAuthService = {
  verifyToken: (token: string) => {
    try {
      return verify(token, JWT_SECRET) as JwtUserPayload;
    } catch (e) {
      return null;
    }
  }
};

// Define ProcessingJob interface
interface ProcessingJob {
  id: string;
  userId?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: any;
  error?: string;
  startedAt: Date;
  completedAt?: Date;
  createdAt?: Date;
  metadata?: Record<string, any>;
}

// File processor implementation with event emission
class FileProcessor extends EventEmitter {
  private jobs = new Map<string, ProcessingJob>();

  getJob(jobId: string): ProcessingJob | null {
    return this.jobs.get(jobId) || null;
  }

  updateJob(jobId: string, updates: Partial<ProcessingJob>): ProcessingJob | null {
    const job = this.getJob(jobId);
    if (!job) return null;
    
    const updatedJob = { ...job, ...updates, updatedAt: new Date() };
    this.jobs.set(jobId, updatedJob);
    return updatedJob;
  }

  createJob(initialData: Partial<ProcessingJob>): ProcessingJob {
    const job: ProcessingJob = {
      id: `job_${Date.now()}`,
      status: 'pending',
      progress: 0,
      startedAt: new Date(),
      ...initialData
    };
    
    this.jobs.set(job.id, job);
    return job;
  }

  deleteJob(jobId: string): boolean {
    return this.jobs.delete(jobId);
  }
}

// Create a singleton instance
const fileProcessor = new FileProcessor();

// Rate limiting for WebSocket messages
const messageRateLimiter = new RateLimiterMemory({
  points: 100, // 100 messages
  duration: 1, // per second per connection
});

// Rate limiting for connection attempts
const connectionRateLimiter = new RateLimiterMemory({
  points: 5, // 5 connection attempts
  duration: 60, // per minute per IP
});

// Define WebSocket message schema
const messageSchema = z.object({
  type: z.enum(['subscribe', 'unsubscribe', 'message', 'ping', 'pong']),
  channel: z.string().optional(),
  payload: z.record(z.string(), z.any()).optional(),
  requestId: z.string().uuid().optional(),
  timestamp: z.number().int().positive().optional(),
  jobId: z.string().optional(),
  error: z.object({
    code: z.string(),
    message: z.string(),
    timestamp: z.number().optional()
  }).optional(),
  data: z.any().optional()
});

type ValidatedMessage = z.infer<typeof messageSchema>;

interface WebSocketClient extends WebSocket {
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  reconnectTimeout: NodeJS.Timeout | null;
  isAlive: boolean;
  lastActivity: number;
  messageQueue: Array<{ data: any; timestamp: number }>;
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
  [key: string]: any; // Index signature for dynamic properties
}

import { WebSocketAuthenticatedRequest } from '../types/auth';

// WebSocket message type is defined by webSocketMessageSchema

export class WebSocketManager {
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
    if (!ws.userId) return;

    const userClients = this.clients.get(ws.userId);
    if (userClients) {
      userClients.delete(ws);
      if (userClients.size === 0) {
        this.clients.delete(ws.userId);
      }
    }

    logger.info('Client disconnected', {
      clientId: ws.clientId,
      userId: ws.userId,
      remainingConnections: this.getTotalConnections()
    });
  }

  private handleSubscribe(ws: WebSocketClient, channel: string): void {
    if (!ws.subscribedChannels) {
      ws.subscribedChannels = new Set();
    }
    ws.subscribedChannels.add(channel);

    logger.debug('Client subscribed to channel', {
      clientId: ws.clientId,
      userId: ws.userId,
      channel
    });
  }

  private handleUnsubscribe(ws: WebSocketClient, channel: string): void {
    if (ws.subscribedChannels) {
      ws.subscribedChannels.delete(channel);

      logger.debug('Client unsubscribed from channel', {
        clientId: ws.clientId,
        userId: ws.userId,
        channel
      });
    }
  }

  // Constants for throttling and rate limiting
  private static readonly MAX_THROTTLE_ENTRIES = 10000; // Maximum number of throttle entries to keep
  private static readonly THROTTLE_CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes
  private static readonly THROTTLE_MAX_AGE = 15 * 60 * 1000; // 15 minutes
  private static readonly PROGRESS_THROTTLE_MS = 100;

  private wss: WebSocketServer | null = null;
  private clients: Map<string, Set<WebSocketClient>> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionAttempts: Map<string, { count: number; lastAttempt: number }> = new Map();
  private messageRates: Map<string, number> = new Map();
  private count: number = 0;
  private lastAttempt: number = 0;
  private fileProcessor: FileProcessor;

  private progressThrottle = new Map<string, number>();
  private throttleCleanupInterval: NodeJS.Timeout | null = null;
  private readonly PROGRESS_THROTTLE_MS = 100;
  private readonly MAX_THROTTLE_ENTRIES = 1000;
  private cleanupCallbacks: Array<() => void> = [];

  constructor(fileProcessor: FileProcessor) {
    this.fileProcessor = fileProcessor;
    this.setupThrottleCleanup();
    this.setupProcessHandlers();
    this.setupFileProcessorListeners();
  }

  private setupThrottleCleanup(): void {
    // Clean up old throttle entries periodically
    this.throttleCleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [key, timestamp] of this.progressThrottle.entries()) {
        if (now - timestamp > WebSocketManager.PROGRESS_THROTTLE_MS * 10) {
          this.progressThrottle.delete(key);
        }
      }

      // Clean up rate limiting
      const nowMs = Date.now();
      const ipAddresses = Array.from(this.connectionAttempts.keys());
      for (const ip of ipAddresses) {
        const attempt = this.connectionAttempts.get(ip);
        if (attempt && (nowMs - attempt.lastAttempt) > MESSAGE_LIMITS.BLOCK_DURATION_MS) {
          this.connectionAttempts.delete(ip);
        }
      }
    }, 60000); // Cleanup every minute
  }

  private setupProcessHandlers(): void {
    // Handle process events for graceful shutdown
    const shutdown = (signal: string) => {
      logger.info(`Received ${signal}, shutting down gracefully...`);
      this.shutdown();
      process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception in WebSocketManager:', error);
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection at:', promise, 'reason:', reason);
    });
  }

  private setupFileProcessorListeners(): void {
    // Setup file processor event listeners
    this.fileProcessor.on('progress', (job: ProcessingJob) => {
      if (job.userId) {
        this.sendToUser(job.userId, {
          type: WebSocketMessageType.JOB_PROGRESS,
          jobId: job.id,
          payload: {
            progress: job.progress,
            status: job.status,
            message: job.status === 'processing' ? 'Processing...' : 'Completed'
          }
        });
      }
    });
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      const disconnectedClients: WebSocketClient[] = [];

      // Check all clients for timeouts
      for (const clientSet of this.clients.values()) {
        for (const client of clientSet) {
          if (now - client.lastActivity > 30000) { // 30 seconds timeout
            if (client.isAlive) {
              client.isAlive = false;
              client.ping();
            } else {
              disconnectedClients.push(client);
            }
          }
        }
      }

      // Clean up disconnected clients
      for (const client of disconnectedClients) {
        client.terminate();
      }
    }, 10000); // Check every 10 seconds
  }


  private sendError(ws: WebSocketClient, error: { code: string; message: string; requestId?: string; details?: any }): void {
    this.sendToClient(ws, {
      type: WebSocketMessageType.ERROR,
      error: {
        code: error.code,
        message: error.message,
        timestamp: Date.now()
      }
    });
  }

  private sendToClient(ws: WebSocketClient, message: WebSocketMessage): void {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        const messageStr = JSON.stringify(message);
        ws.send(messageStr);
      }
    } catch (error) {
      logger.error('Error sending message to client:', {
        error: error instanceof Error ? error.message : String(error),
        clientId: ws.clientId,
        messageType: message?.type
      });
    }
  }

  /**
   * Initialize WebSocket server
   */
  initialize(server: Server): void {
    this.wss = new WebSocketServer({ 
      server,
      path: '/api/v2/dashboard/ws',
      verifyClient: (info, callback) => {
        this.verifyClientAsync(info)
          .then(result => {
            if (result.valid) {
              // Attach user info to request for later use
              const authReq: WebSocketAuthenticatedRequest = {
                userId: result.userId,
                username: result.username || '',
                sessionId: '',
                ipAddress: info.req.socket.remoteAddress || 'unknown',
                token: undefined,
                ...info.req as any
              };
              if (result.userId) {
                authReq.userId = result.userId;
                authReq.username = result.username || ''; // Provide default empty string
                authReq.sessionId = ''; // Set default empty string for sessionId
                authReq.ipAddress = info.req.socket.remoteAddress || 'unknown';
                callback(true);
              } else {
                callback(false, 401, 'Invalid user ID');
              }
            } else {
              logger.warn('WebSocket authentication failed', {
                reason: result.reason,
                ip: info.req.headers['x-forwarded-for'] || info.req.socket.remoteAddress
              });
              callback(false, 401, result.reason || 'Authentication failed');
            }
          })
          .catch(error => {
            logger.error('WebSocket verification error', {
              error: error.message
            });
            callback(false, 500, 'Internal server error');
          });
      }
    });

    this.wss.on('connection', this.handleConnection.bind(this));
    this.startHeartbeat();
    this.setupFileProcessorListeners();

    logger.info('WebSocket server initialized at /api/v1/dashboard/ws');
  }

  /**
   * Verify client connection (authentication) - async version
   */
  private async verifyClientAsync(info: any): Promise<{ 
    valid: boolean; 
    userId?: string; 
    username?: string; 
    sessionId?: string;
    reason?: string; 
  }> {
    // Check connection count per IP
    const ip = info.req.headers['x-forwarded-for'] || info.req.connection.remoteAddress;
    const now = Date.now();
    
    // Initialize or update connection attempt count
    let attempt = this.connectionAttempts.get(ip) || { count: 0, lastAttempt: 0 };
    
    // Reset counter if last attempt was more than block duration ago
    if (now - attempt.lastAttempt > MESSAGE_LIMITS.BLOCK_DURATION_MS) {
      attempt.count = 0;
    }
    
    // Check connection limit
    if (attempt.count >= MESSAGE_LIMITS.MAX_CONNECTIONS_PER_IP) {
      logger.warn(`Connection limit reached for IP: ${ip}`);
      return { 
        valid: false, 
        reason: 'Connection limit exceeded. Please try again later.' 
      };
    }
    
    // Update attempt counter
    attempt.count++;
    attempt.lastAttempt = now;
    this.connectionAttempts.set(ip, attempt);
    
    // Validate token
    const token = this.extractToken(info);
    if (!token) {
      return { 
        valid: false, 
        reason: 'Authentication token is required' 
      };
    }

    try {
      const decoded = await jwtAuthService.verifyToken(token);
      if (!decoded || !decoded.userId) {
        throw new Error('Invalid token payload');
      }
      
      return { 
        valid: true, 
        userId: decoded.userId,
        username: decoded.username,
        sessionId: decoded.sessionId
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('WebSocket authentication failed:', { 
        error: errorMessage,
        ip,
        userAgent: info.req.headers['user-agent']
      });
      return { 
        valid: false, 
        reason: 'Authentication failed. Please log in again.' 
      };
    }
  }

  private async handleMessage(ws: WebSocketClient, data: RawData): Promise<void> {
    try {
      const message = this.parseMessage(data);
      if (!message) {
        throw new Error('Failed to parse message');
      }
      await this.validateMessage(ws, message);
      await this.processMessage(ws, message);
    } catch (error) {
      this.handleMessageError(ws, error);
    }
  }

  private async validateMessage(ws: WebSocketClient, message: WebSocketMessage): Promise<void> {
    // TO DO: Implement message validation logic
  }

  private async processMessage(ws: WebSocketClient, message: WebSocketMessage): Promise<void> {
    // TO DO: Implement message processing logic
  }

  private handleMessageError(ws: WebSocketClient, error: unknown): void {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : undefined;
    
    logger.error('WebSocket message error:', { 
      error: errorMessage,
      stack: errorStack,
      clientId: ws?.clientId || 'unknown'
    });

    try {
      this.sendError(ws, {
        code: 'MESSAGE_PROCESSING_ERROR',
        message: 'Failed to process message',
        details: errorMessage
      });
    } catch (sendError) {
      logger.error('Failed to send error message to client:', {
        error: sendError instanceof Error ? sendError.message : String(sendError),
        clientId: ws?.clientId || 'unknown'
      });
    }
  }

  /**
   * Get total number of connections across all users
   */
  getTotalConnections(): number {
    let count = 0;
    for (const clientSet of this.clients.values()) {
      count += clientSet.size;
    }
    return count;
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    totalConnections: number;
    connectedUsers: number;
    averageConnectionsPerUser: number;
  } {
    const totalConnections = this.getTotalConnections();
    const connectedUsers = this.clients.size;

    return {
      totalConnections,
      connectedUsers,
      averageConnectionsPerUser: connectedUsers > 0 ? totalConnections / connectedUsers : 0
    };
  }

  /**
   * Shutdown WebSocket server
   */
  shutdown(): void {
    logger.info('Shutting down WebSocket server...');

    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.throttleCleanupInterval) {
      clearInterval(this.throttleCleanupInterval);
      this.throttleCleanupInterval = null;
    }

    // Close all client connections
    for (const clientSet of this.clients.values()) {
      for (const client of clientSet) {
        client.terminate();
      }
    }
    this.clients.forEach((clientSet) => {
      clientSet.forEach(ws => {
        ws.close(1001, 'Server shutting down');
      });
    });

    this.clients.clear();

    if (this.wss) {
      this.wss.close();
    }

    logger.info('WebSocket server shutdown completed');
  }

  private extractToken(info: { req: { url: string } }): string | null {
    try {
      const urlParts = info.req.url?.split('?');
      if (!urlParts || urlParts.length < 2) return null;
      
      const params = new URLSearchParams(urlParts[1]);
      return params.get('token');
    } catch (e) {
      logger.error('Failed to extract token:', e);
      return null;
    }
  }

  private handleCustomMessage(ws: WebSocketClient, message: WebSocketMessage): void {
    logger.debug('Received custom message', { 
      type: message.type,
      userId: ws.userId,
      clientId: ws.clientId
    });
    
    // Echo the message back as an example
    this.sendToClient(ws, {
      type: 'message_ack',
      timestamp: Date.now(),
      requestId: message.requestId,
      data: { received: true }
    });
  }

  private handleConnection(ws: WebSocketClient, req: IncomingMessage): void {
    try {
      const clientId = this.generateClientId();
      const authReq: WebSocketAuthenticatedRequest = {
        userId: '',
        username: '',
        sessionId: '',
        ipAddress: req.socket.remoteAddress || 'unknown',
        ...req as any
      };
      
      // Initialize client properties
      ws.clientId = clientId;
      ws.userId = authReq.userId;
      ws.username = authReq.username || '';
      ws.sessionId = authReq.sessionId;
      ws.ipAddress = authReq.ipAddress || '';
      ws.isAlive = true;
      ws.lastActivity = Date.now();
      ws.messageQueue = [];
      ws.maxQueueSize = MESSAGE_LIMITS.MAX_QUEUE_SIZE;
      ws.subscribedChannels = new Set();
      ws.messageCount = 0;
      ws.lastMessageTime = Date.now();

      // Add client to tracking
      const userClients = this.clients.get(ws.userId) || new Set<WebSocketClient>();
      userClients.add(ws);
      this.clients.set(ws.userId, userClients);

      logger.info('New WebSocket connection established', { 
        clientId,
        userId: ws.userId,
        username: ws.username,
        sessionId: ws.sessionId,
        ipAddress: ws.ipAddress
      });

      // Setup ping/pong for connection health
      ws.on('pong', () => {
        ws.isAlive = true;
        ws.lastActivity = Date.now();
      });

      // Handle incoming messages
      ws.on('message', (data: RawData) => {
        this.handleMessage(ws, data).catch(error => {
          logger.error('Error handling WebSocket message:', error);
        });
      });

      // Handle connection close
      ws.on('close', () => {
        this.handleDisconnect(ws);
      });

      // Handle errors
      ws.on('error', (error) => {
        logger.error('WebSocket error:', error);
        this.handleDisconnect(ws);
      });

      // Send welcome message
      this.sendToClient(ws, {
        type: WebSocketMessageType.CONNECTED,
        timestamp: Date.now(),
        payload: {
          clientId,
          userId: ws.userId,
          message: 'Connection established successfully'
        }
      });
    } catch (error) {
      logger.error('Error in WebSocket connection handler:', error);
      ws.terminate();
    }
  }

  private sendToUser(userId: string, message: WebSocketMessage): void {
    const userClients = this.clients.get(userId);
    if (userClients) {
      let sentCount = 0;
      userClients.forEach(ws => {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            this.sendToClient(ws, message);
            sentCount++;
          }
        } catch (error) {
          logger.error('Error sending message to client:', {
            error: error instanceof Error ? error.message : String(error),
            clientId: ws.clientId,
            userId,
            messageType: message.type
          });
        }
      });

      logger.debug('Message sent to user clients', {
        userId,
        messageType: message.type,
        recipients: sentCount,
        totalClients: userClients.size
      });
    }
  }
}

// Create and export a single instance
const webSocketManager = new WebSocketManager(fileProcessor);
export default webSocketManager;

// Add global error handler for uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception in WebSocket manager', {
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined
  });
  // Don't exit the process, let the application handle it
});

// Add global promise rejection handler
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled promise rejection in WebSocket manager', {
    reason,
    promise
  });
});

// Add event listeners for process cleanup
// Handle graceful shutdown
const shutdown = (signal: string) => {
  logger.info(`${signal} received, cleaning up WebSocket manager...`);
  webSocketManager.shutdown();
  process.exit(0);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));