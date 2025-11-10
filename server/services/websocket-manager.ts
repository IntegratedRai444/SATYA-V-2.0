import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';
import { jwtAuthService } from './jwt-auth-service';
import { fileProcessor, ProcessingJob } from './file-processor';
import { logger } from '../config';
import { rateLimitRules } from '../middleware/advanced-rate-limiting';

interface AuthenticatedWebSocket extends WebSocket {
  userId?: number;
  username?: string;
  isAlive?: boolean;
}

interface WebSocketMessage {
  type: string;
  data?: any;
  error?: string;
  timestamp?: number;
  requestId?: string;
  retryCount?: number;
}

class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients: Map<number, Set<AuthenticatedWebSocket>> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionAttempts: Map<string, { count: number; lastAttempt: number }> = new Map();

  /**
   * Initialize WebSocket server
   */
  initialize(server: Server): void {
    this.wss = new WebSocketServer({ 
      server,
      path: '/ws',
      verifyClient: this.verifyClient.bind(this)
    });

    this.wss.on('connection', this.handleConnection.bind(this));
    this.startHeartbeat();
    this.setupFileProcessorListeners();

    logger.info('WebSocket server initialized');
  }

  /**
   * Verify client connection (authentication)
   */
  private async verifyClient(info: any): Promise<boolean> {
    try {
      const ip = info.req.headers['x-forwarded-for'] || info.req.socket.remoteAddress;
      
      // Apply rate limiting
      const rateLimit = rateLimitRules.websocket;
      const now = Date.now();
      const key = `ws:${ip}`;
      
      // Clean up old entries
      this.connectionAttempts.forEach((value, k) => {
        if (now - value.lastAttempt > rateLimit.windowMs) {
          this.connectionAttempts.delete(k);
        }
      });
      
      // Check rate limit
      const attempt = this.connectionAttempts.get(key) || { count: 0, lastAttempt: 0 };
      if (now - attempt.lastAttempt < rateLimit.windowMs && attempt.count >= rateLimit.maxRequests) {
        logger.warn(`WebSocket connection rate limited for IP: ${ip}`);
        info.req.rateLimited = true;
        return false;
      }
      
      // Update attempt count
      this.connectionAttempts.set(key, {
        count: attempt.count + 1,
        lastAttempt: now
      });

      const url = new URL(info.req.url, 'http://localhost');
      const token = url.searchParams.get('token');

      if (!token) {
        return false;
      }

      const payload = await jwtAuthService.verifyToken(token);
      if (!payload) {
        return false;
      }

      // Store user info for later use
      info.req.userId = payload.userId;
      info.req.username = payload.username;

      return true;
    } catch (error) {
      logger.debug('WebSocket authentication failed', {
        error: (error as Error).message
      });
      return false;
    }
  }

  /**
   * Handle new WebSocket connection
   */
  private handleConnection(ws: AuthenticatedWebSocket, req: any): void {
    const userId = req.userId;
    const username = req.username;
    const clientId = req.headers['x-client-id'] || `client_${Date.now()}`;
    const sessionId = req.headers['x-session-id'] || `sess_${Date.now()}`;

    if (!userId) {
      ws.close(1008, 'Authentication required');
      return;
    }

    ws.userId = userId;
    ws.username = username;
    ws.isAlive = true;
    
    // Add client metadata for tracking
    (ws as any).clientId = clientId;
    (ws as any).sessionId = sessionId;
    (ws as any).connectedAt = Date.now();
    (ws as any).lastActivity = Date.now();
    (ws as any).messageQueue = [];

    // Add client to user's connection set
    if (!this.clients.has(userId)) {
      this.clients.set(userId, new Set());
    }
    this.clients.get(userId)!.add(ws);

    logger.info('WebSocket client connected', {
      userId,
      username,
      totalConnections: this.getTotalConnections()
    });

    // Send welcome message
    this.sendToClient(ws, {
      type: 'connected',
      data: {
        message: 'Connected to SatyaAI processing updates',
        userId,
        timestamp: new Date().toISOString()
      }
    });

    // Handle client messages
    ws.on('message', (data) => {
      this.handleMessage(ws, data);
    });

    // Handle client disconnect
    ws.on('close', () => {
      this.handleDisconnect(ws);
    });

    // Handle errors
    ws.on('error', (error) => {
      logger.error('WebSocket error', {
        userId,
        error: error.message
      });
    });

    // Handle pong responses
    ws.on('pong', () => {
      ws.isAlive = true;
    });
  }

  /**
   * Handle incoming messages from clients
   */
  private handleMessage(ws: AuthenticatedWebSocket, data: any): void {
    try {
      const message = JSON.parse(data.toString());
      
      switch (message.type) {
        case 'ping':
          this.sendToClient(ws, { type: 'pong' });
          break;
          
        case 'subscribe_job':
          // Client wants updates for a specific job
          if (message.jobId) {
            this.subscribeToJob(ws, message.jobId);
          }
          break;
          
        case 'unsubscribe_job':
          // Client no longer wants updates for a job
          if (message.jobId) {
            this.unsubscribeFromJob(ws, message.jobId);
          }
          break;
          
        default:
          logger.debug('Unknown WebSocket message type', {
            type: message.type,
            userId: ws.userId
          });
      }
    } catch (error) {
      logger.error('Error handling WebSocket message', {
        error: (error as Error).message,
        userId: ws.userId
      });
    }
  }

  /**
   * Handle client disconnect
   */
  private handleDisconnect(ws: AuthenticatedWebSocket): void {
    if (ws.userId) {
      const userClients = this.clients.get(ws.userId);
      if (userClients) {
        userClients.delete(ws);
        if (userClients.size === 0) {
          this.clients.delete(ws.userId);
        }
      }

      logger.info('WebSocket client disconnected', {
        userId: ws.userId,
        username: ws.username,
        totalConnections: this.getTotalConnections()
      });
    }
  }

  /**
   * Subscribe client to job updates
   */
  private subscribeToJob(ws: AuthenticatedWebSocket, jobId: string): void {
    const job = fileProcessor.getJob(jobId);
    
    if (!job) {
      this.sendToClient(ws, {
        type: 'error',
        error: 'Job not found'
      });
      return;
    }

    // Check if user has permission to view this job
    if (ws.userId !== job.userId) {
      this.sendToClient(ws, {
        type: 'error',
        error: 'Access denied'
      });
      return;
    }

    // Send current job status
    this.sendToClient(ws, {
      type: 'job_status',
      data: {
        jobId: job.id,
        status: job.status,
        progress: job.progress,
        result: job.result,
        error: job.error
      }
    });
  }

  /**
   * Unsubscribe client from job updates
   */
  private unsubscribeFromJob(ws: AuthenticatedWebSocket, jobId: string): void {
    this.sendToClient(ws, {
      type: 'unsubscribed',
      data: { jobId }
    });
  }

  /**
   * Send message to specific client
   */
  private async sendToClient(ws: AuthenticatedWebSocket, message: WebSocketMessage, maxRetries = 3): Promise<boolean> {
    if (!message.timestamp) {
      message.timestamp = Date.now();
    }

    const sendMessage = async (attempt = 0): Promise<boolean> => {
      if (ws.readyState !== WebSocket.OPEN) {
        logger.warn('WebSocket not ready, queuing message', { 
          userId: ws.userId,
          messageType: message.type 
        });
        
        // Add to queue with retry count
        (ws as any).messageQueue.push({
          ...message,
          retryCount: (message.retryCount || 0) + 1
        });
        
        // Try to flush queue if connection recovers
        if ((ws as any).flushTimeout) clearTimeout((ws as any).flushTimeout);
        (ws as any).flushTimeout = setTimeout(() => this.flushMessageQueue(ws), 1000);
        
        return false;
      }

      try {
        ws.send(JSON.stringify(message));
        (ws as any).lastActivity = Date.now();
        return true;
      } catch (error) {
        if (attempt < maxRetries) {
          logger.warn(`Retrying WebSocket message (attempt ${attempt + 1})`, {
            error: (error as Error).message,
            userId: ws.userId,
            messageType: message.type
          });
          await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
          return sendMessage(attempt + 1);
        }
        
        logger.error('Failed to send WebSocket message after retries', {
          error: (error as Error).message,
          userId: ws.userId,
          messageType: message.type,
          retryCount: attempt
        });
        
        return false;
      }
    };

    return sendMessage();
  }
  
  private async flushMessageQueue(ws: AuthenticatedWebSocket): Promise<void> {
    if (!(ws as any).messageQueue?.length) return;
    
    const queue = [...(ws as any).messageQueue];
    (ws as any).messageQueue = [];
    
    for (const message of queue) {
      const sent = await this.sendToClient(ws, message);
      if (!sent) {
        // Re-queue failed messages
        (ws as any).messageQueue.push(message);
      }
      
      // Small delay between messages to prevent overwhelming
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Schedule next flush if queue isn't empty
    if ((ws as any).messageQueue?.length) {
      (ws as any).flushTimeout = setTimeout(() => this.flushMessageQueue(ws), 1000);
    }
  }

  /**
   * Send message to all clients of a user
   */
  private sendToUser(userId: number, message: WebSocketMessage): void {
    const userClients = this.clients.get(userId);
    if (userClients) {
      userClients.forEach(ws => {
        this.sendToClient(ws, message);
      });
    }
  }

  /**
   * Setup file processor event listeners
   */
  private setupFileProcessorListeners(): void {
    fileProcessor.on('jobStarted', (job: ProcessingJob) => {
      this.sendToUser(job.userId, {
        type: 'job_started',
        data: {
          jobId: job.id,
          status: job.status,
          progress: job.progress,
          timestamp: Date.now(),
          estimatedCompletion: this.calculateEstimatedCompletion(job),
          stages: [
            { id: 'queued', status: 'complete', startedAt: job.createdAt },
            { id: 'processing', status: 'active', startedAt: Date.now() },
            { id: 'completed', status: 'pending' }
          ]
        }
      });
    });
    
    fileProcessor.on('jobProgress', (job: ProcessingJob, stage?: string, progress?: number) => {
      const progressUpdate = {
        jobId: job.id,
        status: job.status,
        progress: progress || job.progress,
        stage: stage || 'processing',
        timestamp: Date.now(),
        estimatedCompletion: this.calculateEstimatedCompletion(job),
        metrics: job.metrics || {}
      };
      
      this.sendToUser(job.userId, {
        type: 'job_progress',
        data: progressUpdate
      });
    });
    
    fileProcessor.on('jobStage', (job: ProcessingJob, stage: string) => {
      this.sendToUser(job.userId, {
        type: 'job_stage_update',
        data: {
          jobId: job.id,
          status: job.status,
          stage,
          timestamp: Date.now(),
          message: `Processing stage: ${stage}`
        }
      });
    });
    
    fileProcessor.on('jobMetrics', (job: ProcessingJob, metrics: Record<string, any>) => {
      this.sendToUser(job.userId, {
        type: 'job_metrics',
        data: {
          jobId: job.id,
          metrics,
          timestamp: Date.now()
        }
      });
    });
    
    // Add handler for custom events
    fileProcessor.on('jobEvent', (job: ProcessingJob, event: string, data: any) => {
      this.sendToUser(job.userId, {
        type: `job_${event}`,
        data: {
          jobId: job.id,
          ...data,
          timestamp: Date.now()
        }
      });
    });
  }

  /**
   * Start heartbeat to detect disconnected clients
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      const stats = {
        totalConnections: 0,
        activeConnections: 0,
        inactiveConnections: 0,
        terminatedConnections: 0
      };
      
      this.clients.forEach((clientSet, userId) => {
        clientSet.forEach(ws => {
          stats.totalConnections++;
          
          // Check for stale connections
          const lastActivity = (ws as any).lastActivity || 0;
          const connectionAge = now - ((ws as any).connectedAt || now);
          const inactiveDuration = now - lastActivity;
          
          // Terminate if:
          // 1. Connection is marked as not alive (no pong response)
          // 2. No activity for more than 5 minutes
          // 3. Connection is older than 24 hours (force reconnect)
          if (!ws.isAlive || 
              inactiveDuration > 5 * 60 * 1000 || 
              connectionAge > 24 * 60 * 60 * 1000) {
                
            logger.info('Terminating WebSocket connection', { 
              userId,
              reason: !ws.isAlive ? 'no_heartbeat' : 
                      inactiveDuration > 5 * 60 * 1000 ? 'inactive' : 'max_age_reached',
              connectionAge,
              inactiveDuration,
              clientId: (ws as any).clientId
            });
            
            try {
              ws.terminate();
              stats.terminatedConnections++;
            } catch (error) {
              logger.error('Error terminating WebSocket connection', {
                userId,
                error: (error as Error).message
              });
            }
            return;
          }
          
          stats.activeConnections++;
          ws.isAlive = false;
          
          // Send ping with timestamp
          try {
            ws.ping(JSON.stringify({ 
              type: 'ping',
              timestamp: now,
              connectionId: (ws as any).clientId
            }));
          } catch (error) {
            logger.error('Error sending WebSocket ping', {
              userId,
              error: (error as Error).message
            });
          }
        });
      });
      
      // Log connection stats every 5 minutes
      if (Date.now() % (5 * 60 * 1000) < 1000) {
        logger.info('WebSocket connection statistics', stats);
      }
    }, 30000); // Check every 30 seconds
  }
  
  private calculateEstimatedCompletion(job: ProcessingJob): number | null {
    if (!job.startedAt) return null;
    let total = 0;
    this.clients.forEach(clientSet => {
      total += clientSet.size;
    });
    return total;
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
    }

    // Close all client connections
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
}

// Export singleton instance
export const webSocketManager = new WebSocketManager();

// Graceful shutdown handlers
process.on('SIGTERM', () => {
  webSocketManager.shutdown();
});

process.on('SIGINT', () => {
  webSocketManager.shutdown();
});