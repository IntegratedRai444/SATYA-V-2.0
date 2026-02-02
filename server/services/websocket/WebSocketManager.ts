import { WebSocketServer, WebSocket, RawData } from 'ws';
import { Server, IncomingMessage } from 'http';
import { verify } from 'jsonwebtoken';
import { z } from 'zod';
import { logger } from '../../config/logger';
import { JWT_SECRET_FINAL, WS_MAX_MESSAGES_PER_SECOND, WS_HEARTBEAT_INTERVAL } from '../../config/constants';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { v4 as uuidv4 } from 'uuid';
import { metrics } from '../../monitoring/metrics';
import { URL } from 'url';
import { setInterval, clearInterval } from 'timers';

import { WebSocketMetrics } from './WebSocketMetrics';

// WebSocket message rate limiting
const messageRateLimiter = new RateLimiterMemory({
  points: WS_MAX_MESSAGES_PER_SECOND || 10, // Default to 10 messages per second if not set
  duration: 1, // per second
  keyPrefix: 'ws_msg_' // Add a key prefix for better identification in metrics/storage
});

// Define WebSocket message schema
const messageSchema = z.object({
  type: z.enum(['subscribe', 'unsubscribe', 'message', 'ping', 'pong']),
  channel: z.string().optional(),
  payload: z.record(z.string(), z.unknown()).optional(),
  requestId: z.string().uuid().optional(),
  timestamp: z.number().int().positive().optional(),
});

type ValidatedMessage = z.infer<typeof messageSchema>;

interface TokenPayload {
  userId: string;
  sessionId: string;
  [key: string]: unknown;
}

interface AuthenticatedWebSocket extends WebSocket {
  id: string;
  userId: string;
  sessionId: string;
  ipAddress: string;
  isAlive: boolean;
  lastActivity: number;
  subscribedChannels: Set<string>;
  messageCount: number;
  lastMessageTime: number;
  requestId?: string;
  traceId?: string;
  spanId?: string;
  connectTime: number;
  lastPingTime?: number;
  messageRate: number;
  errorCount: number;
  userAgent?: string;
  clientVersion?: string;
  metadata: Record<string, unknown>;
}

interface WebSocketMessage {
  type: string;
  data?: unknown;
  error?: string;
  timestamp?: number;
  requestId?: string;
  retryCount?: number;
  channel?: string;
  traceId?: string;
  spanId?: string;
}

class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients: Map<string, AuthenticatedWebSocket> = new Map();
  private channelSubscribers: Map<string, Set<string>> = new Map();
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;

  /**
   * Get WebSocket connection statistics
   */
  public getStats() {
    const uniqueUsers = new Set<string>();
    let totalConnections = 0;
    
    // Count unique users and total connections
    this.clients.forEach(client => {
      uniqueUsers.add(client.userId);
      totalConnections++;
    });

    // Count subscribers per channel
    const channelStats: Record<string, number> = {};
    this.channelSubscribers.forEach((subscribers, channel) => {
      channelStats[channel] = subscribers.size;
    });

    return {
      totalConnections,
      connectedUsers: uniqueUsers.size,
      channels: Object.keys(channelStats).length,
      channelStats
    };
  }

  /**
   * Initialize WebSocket server
   */
  initialize(server: Server): void {
    this.wss = new WebSocketServer({ 
      noServer: true,
      clientTracking: true,
    });

    // Setup WebSocket upgrade handler
    server.on('upgrade', async (request, socket, head) => {
      try {
        // Extract token from query parameters
        const token = new URL(request.url || '', `http://${request.headers.host}`).searchParams.get('token');
        
        if (!token) {
          socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
          socket.destroy();
          return;
        }

        // Verify JWT token
        let decoded: TokenPayload;
        try {
          decoded = verify(token, JWT_SECRET_FINAL) as TokenPayload;
        } catch (error) {
          socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
          socket.destroy();
          return;
        }

        // Handle the WebSocket upgrade
        this.wss?.handleUpgrade(request, socket, head, (ws) => {
          this.initializeClient(ws as AuthenticatedWebSocket, decoded, request);
        });
      } catch (error) {
        logger.error('WebSocket upgrade error:', error);
        socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\n');
        socket.destroy();
      }
    });

    // Start heartbeat
    this.startHeartbeat();

    // Handle server close
    process.on('SIGTERM', () => this.cleanup());
    process.on('SIGINT', () => this.cleanup());
  }

  private initializeClient(ws: AuthenticatedWebSocket, tokenPayload: TokenPayload, request: IncomingMessage) {
    const ip = request.headers['x-forwarded-for']?.toString().split(',')[0].trim() || 
              request.socket.remoteAddress || 
              'unknown';
    const clientId = uuidv4();
    const requestId = request.headers['x-request-id']?.toString() || uuidv4();
    const traceId = request.headers['x-trace-id']?.toString() || uuidv4();
    const spanId = uuidv4();
    const userAgent = request.headers['user-agent'] || 'unknown';
    const clientVersion = request.headers['x-client-version']?.toString();
    
    // Initialize client properties with tracing context
    ws.id = clientId;
    ws.userId = tokenPayload.userId;
    ws.sessionId = tokenPayload.sessionId;
    ws.ipAddress = ip;
    ws.isAlive = true;
    ws.lastActivity = Date.now();
    ws.subscribedChannels = new Set();
    ws.messageCount = 0;
    ws.lastMessageTime = 0;
    ws.requestId = requestId;
    ws.traceId = traceId;
    ws.spanId = spanId;
    ws.connectTime = Date.now();
    ws.messageRate = 0;
    ws.errorCount = 0;
    ws.userAgent = userAgent;
    ws.clientVersion = clientVersion;
    ws.metadata = {};

    // Update metrics
    WebSocketMetrics.connectionOpened();
    // Update active connections count
    metrics.websocket.connections.set({ status: 'connected' }, this.clients.size);

    // Log connection
    logger.info('WebSocket client connected', {
      clientId,
      userId: tokenPayload.userId,
      ip,
      userAgent,
      clientVersion,
      traceId,
      spanId,
      requestId,
      timestamp: new Date().toISOString()
    });

    // Add to clients map
    this.clients.set(ws.id, ws);

    // Setup event handlers
    ws.on('message', (data) => this.handleMessage(ws, data));
    ws.on('pong', () => this.handlePong(ws));
    ws.on('close', () => this.handleClose(ws));
    ws.on('error', (error) => this.handleError(ws, error));

    // Send welcome message
    this.send(ws, {
      type: 'connected',
      data: { userId: ws.userId, sessionId: ws.sessionId },
      timestamp: Date.now(),
    });

    logger.info(`Client connected: ${ws.id} (User: ${ws.userId})`);
  }

  private async handleMessage(ws: AuthenticatedWebSocket, message: RawData) {
    const start = process.hrtime();
    try {
      // Rate limiting
      await messageRateLimiter.consume(ws.id, 1);
      
      // Parse the message
      const parsedMessage = this.parseMessage(message);
      
      // Check if message was parsed successfully
      if (!parsedMessage) {
        this.sendError(ws, 'invalid_message', 'Failed to parse message', ws.requestId);
        return;
      }
      
      // Track message received
      WebSocketMetrics.messageReceived(parsedMessage.type || 'unknown');
      
      // Process the message based on type
      await this.routeMessage(ws, parsedMessage);
      
      // Track successful processing
      const [seconds, nanoseconds] = process.hrtime(start);
      WebSocketMetrics.messageProcessed(
        parsedMessage.type || 'unknown',
        seconds + nanoseconds / 1e9
      );
    } catch (error) {
      if (error && typeof error === 'object' && 'msBeforeNext' in error) {
        // Rate limit exceeded
        this.sendError(ws, 'rate_limit_exceeded', 'Too many messages', ws.requestId);
        const rateLimitError = new Error('Rate limit exceeded');
        WebSocketMetrics.errorOccurred('rate_limit', rateLimitError);
      } else {
        this.handleError(ws, error as Error, 'message_processing');
      }
    }
  }

  private handleConnection(ws: WebSocket, request: IncomingMessage) {
    const client = ws as AuthenticatedWebSocket;
    client.id = uuidv4();
    client.isAlive = true;
    client.lastActivity = Date.now();
    client.subscribedChannels = new Set();
    client.messageCount = 0;
    client.errorCount = 0;
    client.connectTime = Date.now();
    client.metadata = {};
    client.requestId = request.headers['x-request-id'] as string || uuidv4();
    client.traceId = request.headers['x-trace-id'] as string || uuidv4();
    client.spanId = uuidv4().split('-')[0];
    client.ipAddress = request.socket.remoteAddress || 'unknown';
    client.userAgent = request.headers['user-agent'] || 'unknown';
    client.clientVersion = request.headers['x-client-version'] as string || 'unknown';
    
    // Track connection
    WebSocketMetrics.connectionOpened();
  }

  private handlePong(ws: AuthenticatedWebSocket) {
    ws.isAlive = true;
    ws.lastActivity = Date.now();
  }

  private handleClose(client: AuthenticatedWebSocket) {
    try {
      // Unsubscribe from all channels
      client.subscribedChannels.forEach(channel => {
        const subscribers = this.channelSubscribers.get(channel);
        if (subscribers) {
          subscribers.delete(client.id);
          if (subscribers.size === 0) {
            this.channelSubscribers.delete(channel);
          }
        }
      });
      
      // Remove client from active connections
      this.clients.delete(client.id);
      
      // Track connection closed
      WebSocketMetrics.connectionClosed();
      
      logger.info(`Client disconnected: ${client.id}`, {
        userId: client.userId,
        sessionId: client.sessionId,
        ipAddress: client.ipAddress,
        duration: Date.now() - client.connectTime,
        messageCount: client.messageCount,
        errorCount: client.errorCount,
        requestId: client.requestId,
        traceId: client.traceId,
      });
    } catch (error) {
      WebSocketMetrics.errorOccurred('close_handler', error as Error);
      logger.error('Error in WebSocket close handler', {
        error: (error as Error).message,
        stack: (error as Error).stack,
        clientId: client.id,
        requestId: client.requestId,
        traceId: client.traceId,
      });
    }
  }

  private startHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      this.clients.forEach((client) => {
        try {
          if (!client.isAlive) {
            logger.warn('Terminating unresponsive WebSocket connection', {
              clientId: client.id,
              userId: client.userId,
              lastActivity: new Date(client.lastActivity).toISOString(),
              durationInactive: now - client.lastActivity,
              requestId: client.requestId,
              traceId: client.traceId
            });
            WebSocketMetrics.errorOccurred('heartbeat_timeout', new Error('Client heartbeat timeout'));
            client.terminate();
            return;
          }
          
          client.isAlive = false;
          client.ping();
          
          // Track heartbeat - using a counter instead of gauge
          metrics.websocket.messages.inc(
            { 
              type: 'heartbeat', 
              status: 'received' 
            },
            1 // Increment by 1
          );
          
        } catch (error) {
          WebSocketMetrics.errorOccurred('heartbeat_error', error as Error);
          logger.error('Error in WebSocket heartbeat', {
            error: (error as Error).message,
            stack: (error as Error).stack,
            clientId: client.id,
            requestId: client.requestId,
            traceId: client.traceId
          });
        }
      });
    }, WS_HEARTBEAT_INTERVAL);
  }

  /**
   * Handle WebSocket errors
   */
  private handleError(client: AuthenticatedWebSocket, error: Error, type: string = 'unknown') {
    client.errorCount++;
    
    // Track the error
    WebSocketMetrics.errorOccurred(type, error);
    
    logger.error(`WebSocket error (${type}): ${error.message}`, {
      error: error.stack,
      clientId: client.id,
      userId: client.userId,
      requestId: client.requestId,
      traceId: client.traceId,
      spanId: client.spanId,
      timestamp: new Date().toISOString()
    });
    
    this.sendError(client, type, error.message);
  }

  /**
   * Parse incoming WebSocket message
   */
  private parseMessage(message: RawData): ValidatedMessage | null {
    try {
      const data = message.toString();
      if (!data) {
        throw new Error('Empty message received');
      }
      
      const parsed = JSON.parse(data);
      const result = messageSchema.safeParse(parsed);
      
      if (!result.success) {
        logger.warn('Invalid WebSocket message format', {
          error: result.error,
          message: data
        });
        return null;
      }
      
      return result.data;
    } catch (error) {
      logger.error('Failed to parse WebSocket message:', {
        error: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
        message: message.toString().substring(0, 1000) // Log first 1000 chars to avoid huge logs
      });
      return null;
    }
  }

  /**
   * Route incoming WebSocket messages to appropriate handlers
   */
  private async routeMessage(ws: AuthenticatedWebSocket, message: ValidatedMessage) {
    if (!message) return;

    try {
      // Update last activity
      ws.lastActivity = Date.now();

      // Handle different message types
      switch (message.type) {
        case 'ping':
          this.sendMessage(ws, { type: 'pong', timestamp: Date.now() });
          break;
          
        case 'subscribe':
          if (message.channel) {
            this.subscribeToChannel(ws, message.channel);
          }
          break;
          
        case 'unsubscribe':
          if (message.channel) {
            this.unsubscribeFromChannel(ws, message.channel);
          }
          break;
          
        case 'message':
          await this.handleCustomMessage(ws, message);
          break;
          
        default:
          logger.warn(`Unknown message type: ${message.type}`);
      }
    } catch (error) {
      logger.error('Error routing WebSocket message:', error);
      this.sendError(ws, 'Failed to process message', 'message_processing_error');
    }
  }

  /**
   * Send message to WebSocket client
   */
  /**
   * Send a message to a WebSocket client
   * @deprecated Use sendMessage instead
   */
  private send(ws: WebSocket, message: WebSocketMessage) {
    this.sendMessage(ws as AuthenticatedWebSocket, message);
  }

  /**
   * Send a message to a WebSocket client
   */
  private sendMessage(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        const messageString = JSON.stringify({
          ...message,
          timestamp: message.timestamp || Date.now(),
          requestId: message.requestId || ws.requestId,
          traceId: message.traceId || ws.traceId,
          spanId: message.spanId || ws.spanId
        });
        
        ws.send(messageString);
        
        // Track message metrics with correct label format
        metrics.websocket.messages.inc(
          { 
            type: message.type || 'unknown',
            status: 'sent' 
          },
          1 // Increment by 1
        );
        
        ws.messageCount = (ws.messageCount || 0) + 1;
      } catch (error) {
        logger.error('Failed to send WebSocket message:', error);
        this.handleError(ws, error as Error, 'send_message_failed');
      }
    }
  }

  private sendError(
    ws: AuthenticatedWebSocket, 
    message: string, 
    details?: unknown, 
    requestId?: string
  ) {
    const errorResponse: WebSocketMessage = {
      type: 'error',
      error: message,
      timestamp: Date.now(),
      requestId: requestId || ws.requestId,
    };
    
    if (details) {
      errorResponse.data = { details };
    }
    
    // Log the error
    logger.error('WebSocket error', {
      error: message,
      details,
      clientId: ws.id,
      userId: ws.userId,
      requestId: errorResponse.requestId,
      // Include tracing context in the log instead of the response
      traceId: ws.traceId,
      spanId: ws.spanId,
      timestamp: new Date().toISOString()
    });
    
    // Update metrics
    WebSocketMetrics.errorOccurred('application', new Error(message));
    
    // Send the error response if the connection is still open
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(errorResponse));
      } catch (error) {
        logger.error('Failed to send WebSocket error response', {
          error: error instanceof Error ? error.message : String(error),
          clientId: ws.id,
          userId: ws.userId,
          requestId: errorResponse.requestId,
          traceId: errorResponse.traceId,
          spanId: errorResponse.spanId,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Subscribe client to a channel
   */
  private subscribeToChannel(ws: AuthenticatedWebSocket, channel: string) {
    if (!ws.subscribedChannels.has(channel)) {
      ws.subscribedChannels.add(channel);
      
      if (!this.channelSubscribers.has(channel)) {
        this.channelSubscribers.set(channel, new Set());
      }
      
      this.channelSubscribers.get(channel)?.add(ws.id);
      
      // Send confirmation
      this.sendMessage(ws, {
        type: 'subscribed',
        channel,
        timestamp: Date.now()
      });
      
      logger.info(`Client ${ws.id} subscribed to channel: ${channel}`, {
        userId: ws.userId,
        channel,
        requestId: ws.requestId,
        traceId: ws.traceId
      });
    }
  }

  /**
   * Unsubscribe client from a channel
   */
  private unsubscribeFromChannel(ws: AuthenticatedWebSocket, channel: string) {
    if (ws.subscribedChannels.has(channel)) {
      ws.subscribedChannels.delete(channel);
      this.channelSubscribers.get(channel)?.delete(ws.id);
      
      // Clean up empty channels
      if (this.channelSubscribers.get(channel)?.size === 0) {
        this.channelSubscribers.delete(channel);
      }
      
      // Send confirmation
      this.sendMessage(ws, {
        type: 'unsubscribed',
        channel,
        timestamp: Date.now()
      });
      
      logger.info(`Client ${ws.id} unsubscribed from channel: ${channel}`, {
        userId: ws.userId,
        channel,
        requestId: ws.requestId,
        traceId: ws.traceId
      });
    }
  }

  /**
   * Handle custom message types
   */
  private async handleCustomMessage(ws: AuthenticatedWebSocket, message: ValidatedMessage) {
    try {
      // Implement custom message handling logic here
      // Example: Broadcast to all clients in the same channel
      if (message.channel) {
        await this.broadcastToChannel(message.channel, {
          type: 'message',
          channel: message.channel,
          data: message.payload,
          timestamp: Date.now(),
          requestId: ws.requestId,
          traceId: ws.traceId,
          spanId: ws.spanId
        });
      }
    } catch (error) {
      logger.error('Error handling custom message:', error);
      this.sendError(ws, 'Failed to process custom message', 'custom_message_error');
    }
  }

  /**
   * Broadcast message to all clients in a channel
   */
  private async broadcastToChannel(channel: string, message: WebSocketMessage) {
    const subscribers = this.channelSubscribers.get(channel);
    if (!subscribers) return;

    const messageString = JSON.stringify(message);
    
    subscribers.forEach(clientId => {
      const client = this.clients.get(clientId);
      if (client && client.readyState === WebSocket.OPEN) {
        try {
          client.send(messageString);
          // Track broadcast messages with correct label format
          metrics.websocket.messages.inc({
            type: 'broadcast',
            status: 'sent'
          });
          
          // Log the broadcast for debugging
          logger.debug('Broadcast message sent', {
            channel,
            clientId: client.id,
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          logger.error('Error broadcasting message:', error);
        }
      }
    });
  }

  /**
   * Clean up resources
   */
  private cleanup() {
    // Close all connections
    this.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.close(1001, 'Server shutting down');
      }
    });

    // Clear interval
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    // Clear data structures
    this.clients.clear();
    this.channelSubscribers.clear();
    this.wss?.close();
  }
}

// Lazy initialization wrapper
class WebSocketService {
  private static instance: WebSocketService;
  private manager: WebSocketManager | null = null;
  private server: Server | null = null;

  private constructor() {}

  public static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  public initialize(server: Server): void {
    if (this.manager) return;
    
    this.server = server;
    this.manager = new WebSocketManager();
    this.manager.initialize(server);
  }

  public getManager(): WebSocketManager {
    if (!this.manager) {
      throw new Error('WebSocketManager not initialized. Call initialize() first.');
    }
    return this.manager;
  }

  public isInitialized(): boolean {
    return !!this.manager;
  }
}

// Export singleton instance
export const webSocketService = WebSocketService.getInstance();
