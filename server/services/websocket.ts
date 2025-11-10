import { Server as HttpServer } from 'http';
import { Server as WsServer, Socket } from 'socket.io';
import { logger } from '../config/logger';
import { rateLimitRules } from '../middleware/advanced-rate-limiting';

class WebSocketService {
  private io: WsServer;
  private connectedClients: Map<string, Socket> = new Map();

  initialize(server: HttpServer): void {
    this.io = new WsServer(server, {
      cors: {
        origin: process.env.FRONTEND_URL || 'http://localhost:3000',
        methods: ['GET', 'POST']
      },
      maxHttpBufferSize: 1e8 // 100MB for large file transfers
    });

    // Apply connection rate limiting
    this.io.use((socket, next) => {
      const ip = socket.handshake.address;
      const key = `ws:${ip}`;
      
      const rateLimit = rateLimitRules.websocket;
      const entry = this.getRateLimitEntry(key, rateLimit.windowMs);
      
      if (entry.count >= rateLimit.maxRequests) {
        logger.warn(`WebSocket connection rate limited: ${ip}`);
        return next(new Error('Too many connection attempts. Please try again later.'));
      }
      
      entry.count++;
      next();
    });

    this.io.on('connection', (socket: Socket) => {
      const clientId = socket.id;
      this.connectedClients.set(clientId, socket);
      
      logger.info(`Client connected: ${clientId}`);
      
      // Handle analysis progress updates
      socket.on('analysis:progress', (data: { jobId: string; progress: number }) => {
        this.emitToRoom(data.jobId, 'analysis:progress', data);
      });
      
      // Handle job completion
      socket.on('analysis:complete', (data: { jobId: string; result: any }) => {
        this.emitToRoom(data.jobId, 'analysis:complete', data);
      });
      
      // Handle errors
      socket.on('analysis:error', (data: { jobId: string; error: string }) => {
        this.emitToRoom(data.jobId, 'analysis:error', data);
      });
      
      // Join a room for specific job updates
      socket.on('join', (room: string) => {
        socket.join(room);
        logger.debug(`Client ${clientId} joined room: ${room}`);
      });
      
      // Leave a room
      socket.on('leave', (room: string) => {
        socket.leave(room);
        logger.debug(`Client ${clientId} left room: ${room}`);
      });
      
      socket.on('disconnect', () => {
        this.connectedClients.delete(clientId);
        logger.info(`Client disconnected: ${clientId}`);
      });
    });
    
    logger.info('WebSocket server initialized');
  }
  
  private getRateLimitEntry(key: string, windowMs: number): { count: number; resetTime: number } {
    const now = Date.now();
    const entry = (this as any).store.get(key) || { count: 0, resetTime: now + windowMs };
    
    if (now > entry.resetTime) {
      entry.count = 0;
      entry.resetTime = now + windowMs;
    }
    
    (this as any).store.set(key, entry);
    return entry;
  }
  
  emitToRoom(room: string, event: string, data: any): void {
    this.io.to(room).emit(event, data);
  }
  
  getIO(): WsServer {
    if (!this.io) {
      throw new Error('WebSocket server not initialized');
    }
    return this.io;
  }
}

export const webSocketService = new WebSocketService();
