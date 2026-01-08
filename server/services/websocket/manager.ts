import { Server as HttpServer } from 'http';
import { WebSocket, WebSocketServer as WSServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../../config/logger';
import { verifyToken } from '../auth';

interface WebSocketClient extends WebSocket {
  id: string;
  userId?: string;
  isAlive: boolean;
  lastPing?: number;
  metadata?: Record<string, any>;
  ip?: string;
  userAgent?: string;
  connectedAt?: string;
  subscriptions?: string[];
}


export class WebSocketManager {
  private wss: WSServer;
  private clients: Set<WebSocketClient> = new Set();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private readonly HEARTBEAT_INTERVAL = 30000; // 30 seconds
  private readonly PING_TIMEOUT = 10000; // 10 seconds
  private readonly MAX_CONNECTIONS = 1000;
  private readonly MAX_CONNECTIONS_PER_IP = 10;

  constructor(server: HttpServer) {
    this.wss = new WSServer({ 
      server,
      clientTracking: true,
      verifyClient: this.verifyClient.bind(this),
      maxPayload: 1024 * 1024, // 1MB max payload
      perMessageDeflate: {
        zlibDeflateOptions: {
          chunkSize: 1024,
          memLevel: 7,
          level: 3
        },
        zlibInflateOptions: {
          chunkSize: 10 * 1024
        },
        clientNoContextTakeover: true,
        serverNoContextTakeover: true,
        serverMaxWindowBits: 10,
        concurrencyLimit: 10,
        threshold: 1024
      }
    });

    this.setupEventHandlers();
    this.setupHeartbeat();
    
    logger.info('WebSocket server initialized with enhanced security settings');
  }

  private setupEventHandlers() {
    this.wss.on('connection', (ws: WebSocketClient, req) => {
      const ip = req.socket.remoteAddress || 'unknown';
      const userAgent = req.headers['user-agent'] || 'unknown';

      // Set client metadata
      ws.id = uuidv4();
      ws.isAlive = true;
      ws.ip = ip;
      ws.userAgent = userAgent;
      ws.connectedAt = new Date().toISOString();

      // Add rate limiting and connection limits
      if (this.clients.size >= this.MAX_CONNECTIONS) {
        logger.warn(`Connection rejected: Maximum connections (${this.MAX_CONNECTIONS}) reached`);
        ws.close(1013, 'Server busy');
        return;
      }

      if (this.getClientCountByIp(ip) >= this.MAX_CONNECTIONS_PER_IP) {
        logger.warn(`Connection rejected: Too many connections from IP ${ip}`);
        ws.close(1008, 'Too many connections from your IP');
        return;
      }

      this.clients.add(ws);

      // Setup message handler with rate limiting
      let lastMessageTime = 0;
      const MESSAGE_RATE_LIMIT = 100; // ms between messages

      ws.on('message', (data: string) => {
        try {
          const now = Date.now();
          if (now - lastMessageTime < MESSAGE_RATE_LIMIT) {
            logger.warn(`Rate limit exceeded for client ${ws.id}`);
            this.sendError(ws, 'Rate limit exceeded', 429);
            return;
          }
          lastMessageTime = now;

          const message = JSON.parse(data);
          this.handleMessage(ws, message);
        } catch (error) {
          logger.error(`Error processing message from ${ws.id}:`, error);
          this.sendError(ws, 'Invalid message format');
        }
      });

      // Heartbeat handlers
      ws.on('pong', () => {
        ws.isAlive = true;
        ws.lastPing = Date.now();
      });

      // Close handler with cleanup
      ws.on('close', (code, reason) => {
        this.handleClose(ws, code, reason.toString());
      });

      // Error handler
      ws.on('error', (error) => {
        logger.error(`WebSocket error for client ${ws.id}:`, error);
        this.handleClose(ws, 1011, 'Internal server error');
      });

      logger.info(`Client connected: ${ws.id} (IP: ${ip}, User-Agent: ${userAgent})`);

      // Send connection established message
      this.send(ws, {
        type: 'connection_established',
        clientId: ws.id,
        timestamp: Date.now(),
        maxMessageSize: '1MB',
        heartbeatInterval: this.HEARTBEAT_INTERVAL / 1000 + 's'
      });

      // Set up initial ping
      ws.ping();
    });

    // Handle server errors
    this.wss.on('error', (error) => {
      logger.error('WebSocket server error:', error);
    });
  }

  private async verifyClient(
    info: { origin: string; secure: boolean; req: any },
    callback: (res: boolean, code?: number, message?: string) => void
  ) {
    try {
      // Check total connections
      if (this.clients.size >= this.MAX_CONNECTIONS) {
        logger.warn('Connection rejected: Maximum connections reached');
        return callback(false, 429, 'Too many connections');
      }

      // Get client IP
      const ip = info.req.socket.remoteAddress;

      // Check IP-based connection limits
      if (ip && this.getClientCountByIp(ip) >= this.MAX_CONNECTIONS_PER_IP) {
        logger.warn(`Connection rejected: Too many connections from IP ${ip}`);
        return callback(false, 429, 'Too many connections from your IP');
      }

      // Verify authentication token
      const token = this.getTokenFromRequest(info.req);
      if (!token) {
        return callback(false, 401, 'Authentication required');
      }

      // Verify token
      const decoded = await verifyToken(token);
      if (!decoded) {
        return callback(false, 403, 'Invalid or expired token');
      }

      // Add rate limiting info to request
      info.req.rateLimit = {
        ip,
        userAgent: info.req.headers['user-agent']
      };

      // Store user info for later use
      info.req.user = decoded;

      // Connection accepted
      callback(true);

    } catch (error) {
      logger.error('WebSocket verification error:', error);
      callback(false, 500, 'Internal server error');
    }
  }

  private getTokenFromRequest(req: any): string | null {
    // Check Authorization header
    const authHeader = req.headers['authorization'];
    if (authHeader && authHeader.startsWith('Bearer ')) {
      return authHeader.split(' ')[1];
    }

    // Check query parameter
    if (req.url) {
      const url = new URL(req.url, `http://${req.headers.host}`);
      return url.searchParams.get('token');
    }

    return null;
  }

  private handleMessage(ws: WebSocketClient, message: any) {
    // Validate message structure
    if (typeof message !== 'object' || message === null) {
      this.sendError(ws, 'Invalid message format');
      return;
    }

    // Log message for debugging (without sensitive data)
    const logMessage = { ...message };
    if (logMessage.token) logMessage.token = '***';
    if (logMessage.password) logMessage.password = '***';

    logger.debug(`Message from ${ws.id}:`, logMessage);

    // Handle different message types
    try {
      switch (message.type) {
        case 'ping':
          this.send(ws, {
            type: 'pong',
            timestamp: Date.now(),
            serverTime: new Date().toISOString()
          });
          break;

        case 'subscribe':
          this.handleSubscribe(ws, message);
          break;

        case 'unsubscribe':
          this.handleUnsubscribe(ws, message);
          break;

        case 'echo':
          // Simple echo service for testing
          this.send(ws, {
            type: 'echo',
            timestamp: Date.now(),
            data: message.data
          });
          break;

        default:
          this.sendError(ws, 'Unknown message type', 4004);
      }
    } catch (error) {
      logger.error(`Error handling message from ${ws.id}:`, error);
      this.sendError(ws, 'Internal server error', 5001);
    }
  }
  private getClientCountByIp(ip: string): number {
    let count = 0;
    for (const client of this.clients) {
      if (client.ip === ip) {
        count++;
      }
    }
    return count;
  }

private handleSubscribe(ws: WebSocketClient, data: any) {
  try {
    if (!data.channel) {
      throw new Error('Channel is required');
    }
    
    // Initialize subscriptions array if it doesn't exist
    if (!ws.subscriptions) {
      ws.subscriptions = [];
    }
    
    // Check subscription limit (e.g., max 10 subscriptions per client)
    const MAX_SUBSCRIPTIONS = 10;
    if (ws.subscriptions.length >= MAX_SUBSCRIPTIONS) {
      throw new Error(`Maximum of ${MAX_SUBSCRIPTIONS} subscriptions allowed`);
    }
    
    // Add to subscriptions if not already subscribed
    if (!ws.subscriptions.includes(data.channel)) {
      ws.subscriptions.push(data.channel);
      logger.debug(`Client ${ws.id} subscribed to channel: ${data.channel}`);
    }
    
    this.send(ws, {
      type: 'subscription_success',
      channel: data.channel,
      subscriptionCount: ws.subscriptions.length
    });
  } catch (error) {
    this.sendError(ws, error instanceof Error ? error.message : 'Unknown error');
  }
}

  private setupHeartbeat() {
    // Clear existing interval if any
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    // Set up new interval
    this.heartbeatInterval = setInterval(() => {
      this.clients.forEach((ws) => {
        // Check if client is still alive
        if (!ws.isAlive) {
          logger.warn(`Client ${ws.id} did not respond to ping, terminating connection`);
          this.handleClose(ws, 4001, 'No ping response');
          return;
        }

        // Mark as not alive and send ping
        ws.isAlive = false;
        ws.ping();
      });
    }, this.HEARTBEAT_INTERVAL);
  }


  private send(ws: WebSocketClient, data: any) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }

  private sendError(ws: WebSocketClient, message: string, code: number = 4000) {
    const errorResponse = {
      type: 'error',
      error: {
        code,
        message,
        timestamp: new Date().toISOString(),
        requestId: `${ws.id}-${Date.now()}`
      }
    };

    // Log the error
    logger.warn(`Sending error to ${ws.id}:`, {
      code,
      message,
      ip: ws.ip,
      userAgent: ws.userAgent
    });

    // Send the error response
    this.send(ws, errorResponse);
  }

  private handleClose(ws: WebSocketClient, code?: number, reason?: string) {
    // Remove from clients set
    this.clients.delete(ws);
    
    // Log disconnection
    logger.info(`Client disconnected: ${ws.id} (Code: ${code}, Reason: ${reason || 'No reason provided'})`);
    
    // Clean up any resources
    if (ws.readyState === WebSocket.OPEN) {
      ws.terminate();
    }
  }

  private handleUnsubscribe(ws: WebSocketClient, data: any) {
    try {
      // Validate unsubscription request
      if (!data.channel || typeof data.channel !== 'string') {
        return this.sendError(ws, 'Invalid channel', 4003);
      }

      // Remove from subscriptions if it exists
      if (ws.subscriptions?.includes(data.channel)) {
        ws.subscriptions = ws.subscriptions.filter(ch => ch !== data.channel);
        logger.debug(`Client ${ws.id} unsubscribed from ${data.channel}`);
      }

      // Send acknowledgment
      this.send(ws, {
        type: 'unsubscription_ack',
        channel: data.channel,
        success: true,
        timestamp: Date.now(),
        subscriptionCount: ws.subscriptions?.length || 0
      });
    } catch (error) {
      this.sendError(ws, error instanceof Error ? error.message : 'Failed to process unsubscribe request');
    }
  }

}

export default WebSocketManager;
