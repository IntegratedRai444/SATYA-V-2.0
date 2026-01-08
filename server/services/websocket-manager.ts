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

// Message size limit (1MB)
const MAX_MESSAGE_SIZE = 1024 * 1024;

// Message rate limiting (messages per second per connection)
const MESSAGE_RATE_LIMIT = 10;

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

interface AuthenticatedRequest extends IncomingMessage {
  userId: string;
  username?: string;
  sessionId: string;
  ipAddress: string;
  token?: string;
}

// WebSocket message type is defined by webSocketMessageSchema

// Constants for throttling and rate limiting
// Moved to class properties to avoid redeclaration
const MAX_THROTTLE_ENTRIES = 10000; // Maximum number of throttle entries to keep
const THROTTLE_CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes
const THROTTLE_MAX_AGE = 15 * 60 * 1000; // 15 minutes

class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients: Map<string, Set<WebSocketClient>> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionAttempts: Map<string, { count: number; lastAttempt: number }> = new Map();
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
        if (now - timestamp > this.PROGRESS_THROTTLE_MS * 10) {
          this.progressThrottle.delete(key);
        }
      }
    }, 5 * 60 * 1000); // Every 5 minutes
  }

  private setupProcessHandlers(): void {
    // Handle process events for graceful shutdown
    const shutdown = () => this.shutdown();
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
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
    const emitter = this.fileProcessor as unknown as NodeJS.EventEmitter;
    
    emitter.on('jobProgress', (job: ProcessingJob) => {
      if (job.userId) {
        this.sendToUser(job.userId, {
          type: WebSocketMessageType.JOB_PROGRESS,
          jobId: job.id,
          payload: {
            progress: job.progress,
            status: job.status
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

  private handleConnection(ws: WebSocketClient): void {
    ws.isAlive = true;
    ws.lastActivity = Date.now();
    
    // Add ping/pong handlers
    ws.on('pong', () => {
      ws.isAlive = true;
      ws.lastActivity = Date.now();
    });
    
    // Add message handler
    ws.on('message', (data: RawData) => {
      this.handleMessage(ws, data).catch(error => {
        logger.error('Error handling WebSocket message:', error);
      });
    });
    
    // Add error handler
    ws.on('error', (error) => {
      logger.error('WebSocket error:', error);
      ws.terminate();
    });
    
    // Add close handler
    ws.on('close', () => {
      // Clean up client from tracking
      if (ws.userId) {
        const clients = this.clients.get(ws.userId);
        if (clients) {
          clients.delete(ws);
          if (clients.size === 0) {
            this.clients.delete(ws.userId);
          }
        }
      }
    });
  }

  private sendError(ws: WebSocketClient, error: { code: string; message: string }): void {
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
        const messageStr = JSON.stringify({
          ...message,
          timestamp: message.timestamp || Date.now()
        });
        ws.send(messageStr);
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Error sending message to client:', errorMessage);
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
              const authReq = info.req as AuthenticatedRequest;
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
  private async verifyClientAsync(info: any): Promise<{ valid: boolean; userId?: string; username?: string; reason?: string }> {
    const req = info.req as AuthenticatedRequest;
    try {
      const ip = info.req.headers['x-forwarded-for'] || info.req.socket.remoteAddress;
      
      // Apply rate limiting
      const wsRateLimit = rateLimitRules.websocket;
      const now = Date.now();
      const key = `ws:${ip}`;
      
      // Clean up old entries
      this.connectionAttempts.forEach((value, k) => {
        if (now - value.lastAttempt > wsRateLimit.windowMs) {
          this.connectionAttempts.delete(k);
        }
      });
      
      // Extract token from query parameter
      const token = extractTokenFromQuery(info.req.url || '');

      if (!token) {
        logger.debug('WebSocket connection rejected: No token provided');
        return { valid: false, reason: 'No authentication token' };
      }

      // Verify token
      const payload = await jwtAuthService.verifyToken(token);
      if (!payload || !payload.userId) {
        logger.debug('WebSocket connection rejected: Invalid token or missing userId');
        return { valid: false, reason: 'Invalid or expired token' };
      }

      logger.debug('WebSocket authentication successful', {
        userId: payload.userId,
        username: payload.username || 'unknown'
      });

      return {
        valid: true,
        userId: payload.userId,
        username: payload.username || undefined
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('WebSocket authentication error', {
        error: errorMessage
      });
      return { valid: false, reason: 'Authentication failed' };
    }
  };

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

  private async handleMessage(ws: WebSocketClient, data: RawData): Promise<void> {
    try {
      // Check message size
      const messageStr = data.toString();
      if (messageStr.length > MAX_MESSAGE_SIZE) {
        this.sendError(ws, {
          code: 'message_too_large',
          message: 'Message size exceeds limit'
        });
        return;
      }

      // Rate limit message processing
      try {
        await messageRateLimiter.consume(ws.clientId);
      } catch (rateLimiterRes) {
        this.sendError(ws, {
          code: 'rate_limit_exceeded',
          message: 'Too many messages. Please slow down.'
        });
        return;
      }

      // Parse and validate message
      let message: WebSocketMessage;
      try {
        const parsed = JSON.parse(messageStr);
        // Basic validation since we're not using Zod schema anymore
        if (!parsed || typeof parsed !== 'object' || !parsed.type) {
          throw new Error('Invalid message format');
        }
        message = parsed as WebSocketMessage;
      } catch (error) {
        this.sendError(ws, {
          code: 'invalid_message',
          message: 'Invalid message format'
        });
        return;
      }

      // Handle custom message types here
      if (message.type === 'ping') {
        this.sendToClient(ws, { 
          type: 'pong', 
          timestamp: Date.now() 
        });
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorStack = error instanceof Error ? error.stack : undefined;
      logger.error('Error handling WebSocket message', { 
        error: errorMessage,
        stack: errorStack
      });
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

  /**
   * Send message to all clients of a user
   */
  private sendToUser(userId: string, message: WebSocketMessage): void {
    const userClients = this.clients.get(userId);
    if (userClients) {
      userClients.forEach(ws => {
        this.sendToClient(ws, message);
      });
    }
  }
}

// Create a singleton instance
const webSocketManager = new WebSocketManager(fileProcessor);

export { webSocketManager as default, WebSocketManager };

// Add global error handler for uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception in WebSocket manager', {
    error: error.message,
    stack: error.stack
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
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, cleaning up WebSocket manager...');
  webSocketManager.shutdown();
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, cleaning up WebSocket manager...');
  webSocketManager.shutdown();
});