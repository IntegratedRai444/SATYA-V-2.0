/**
 * Enhanced WebSocket Service
 * Provides reliable WebSocket connections with exponential backoff,
 * message queuing, and automatic reconnection
 */

import { getAuthToken } from './auth';
import logger from '../lib/logger';
import type {
  WebSocketMessage,
  ConnectionState,
  MessageHandler,
  EventType,
  QueuedMessage,
} from '../types/websocket';

// Re-export types for backward compatibility
export type {
  WebSocketMessage,
  MessageHandler,
  EventType,
  NotificationMessage,
  ScanUpdateMessage,
  SystemAlertMessage,
} from '../types/websocket';

interface WebSocketOptions {
  maxReconnectAttempts?: number;
  initialReconnectDelay?: number;
  maxReconnectDelay?: number;
  backoffMultiplier?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  debug?: boolean;
}

class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private eventHandlers: Map<EventType, Set<Function>> = new Map();

  // Enhanced reconnection with exponential backoff
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private initialReconnectDelay: number;
  private maxReconnectDelay: number;
  private backoffMultiplier: number;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  // Message queue for offline messages
  private messageQueue: QueuedMessage[] = [];
  private maxQueueSize: number;
  private isProcessingQueue = false;

  // Heartbeat mechanism
  private heartbeatInterval: number;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastHeartbeatTime: number | null = null;
  private missedHeartbeats = 0;
  private readonly MAX_MISSED_HEARTBEATS = 3;

  // Connection state
  private connectionState: ConnectionState = 'disconnected';
  private isAuthenticated = false;
  constructor(options: WebSocketOptions = {}) {
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
    this.initialReconnectDelay = options.initialReconnectDelay || 1000;
    this.maxReconnectDelay = options.maxReconnectDelay || 30000;
    this.backoffMultiplier = options.backoffMultiplier || 2;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.maxQueueSize = options.messageQueueSize || 100;

    // Initialize event handlers
    (['connected', 'disconnected', 'error', 'reconnecting', 'message', 'scan_update', 'notification', 'system_alert', 'dashboard_update'] as EventType[]).forEach(event => {
      this.eventHandlers.set(event, new Set());
    });

    logger.debug('WebSocket service initialized', { options });
  }

  public static getInstance(options?: WebSocketOptions): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(options);
    }
    return WebSocketService.instance;
  }

  public async connect(token?: string): Promise<void> {
    if (this.socket && this.connectionState === 'connected') {
      logger.debug('WebSocket already connected');
      return;
    }

    if (this.socket) {
      this.disconnect();
    }

    this.connectionState = 'connecting';
    const authToken = token || (await getAuthToken());

    if (!authToken) {
      const error = new Error('No authentication token available');
      logger.error('WebSocket connection failed', error);
      this.emit('error', error);
      this.connectionState = 'disconnected';
      return;
    }

    try {
      const wsUrl = this.getWebSocketUrl(authToken);
      logger.debug('Connecting to WebSocket', { url: wsUrl.replace(/token=[^&]+/, 'token=***') });

      this.socket = new WebSocket(wsUrl);
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);
    } catch (error) {
      logger.error('WebSocket connection error', error as Error);
      this.connectionState = 'disconnected';
      this.emit('error', error);
      this.handleReconnect();
    }
  }

  private getWebSocketUrl(token: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    // Use backend server host (localhost:3000) instead of Vite dev server
    const host = process.env.REACT_APP_WS_HOST || 'localhost:3000';
    const path = process.env.REACT_APP_WS_PATH || '/ws';
    return `${protocol}${host}${path}?token=${encodeURIComponent(token)}`;
  }

  private handleOpen() {
    this.reconnectAttempts = 0;
    this.connectionState = 'connected';
    this.isAuthenticated = true;
    this.missedHeartbeats = 0;

    logger.info('WebSocket connected successfully');
    this.startHeartbeat();
    this.emit('connected');

    // Process queued messages
    this.processMessageQueue();
  }

  private handleMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data) as WebSocketMessage;

      // Handle heartbeat/pong response
      if (data.type === 'heartbeat' || data.type === 'pong') {
        this.lastHeartbeatTime = Date.now();
        this.missedHeartbeats = 0;
        logger.debug('Heartbeat received');
        return;
      }

      // Handle authentication response
      if (data.type === 'auth_response') {
        this.isAuthenticated = data.payload.authenticated;
        if (!data.payload.authenticated) {
          const error = new Error(data.payload.error || 'Authentication failed');
          logger.error('WebSocket authentication failed', error);
          this.emit('error', error);
          this.disconnect();
        } else {
          logger.info('WebSocket authenticated');
          this.emit('connected');
        }
        return;
      }

      // Emit generic message event
      this.emit('message', data);

      // Emit type-specific events
      if (data.type === 'scan_update') {
        this.emit('scan_update', data);
      } else if (data.type === 'notification') {
        this.emit('notification', data);
      } else if (data.type === 'system_alert') {
        this.emit('system_alert', data);
      } else if (data.type === 'dashboard_update') {
        this.emit('dashboard_update', data);
      }

      // Notify all message handlers
      this.messageHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (handlerError) {
          logger.error('Error in message handler', handlerError as Error);
        }
      });
    } catch (error) {
      logger.error('Error handling WebSocket message', error as Error);
      this.emit('error', error instanceof Error ? error : new Error('Error parsing WebSocket message'));
    }
  }

  private handleClose(event: CloseEvent) {
    this.stopHeartbeat();
    this.isAuthenticated = false;
    this.connectionState = 'disconnected';

    logger.info('WebSocket disconnected', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
    });

    this.emit('disconnected', event);

    // Only attempt to reconnect if the close was unexpected
    if (event.code !== 1000 && event.code !== 1001) { // 1000 = normal, 1001 = going away
      this.handleReconnect();
    }
  }

  private handleError(error: Event) {
    logger.error('WebSocket error occurred', error as any);
    this.emit('error', error);
  }

  /**
   * Handle reconnection with exponential backoff
   */
  private handleReconnect() {
    // Clear any existing reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      const error = new Error(`Max reconnection attempts (${this.maxReconnectAttempts}) reached`);
      logger.error('WebSocket reconnection failed', error);
      this.emit('error', error);
      return;
    }

    this.reconnectAttempts++;

    // Calculate delay with exponential backoff
    const delay = Math.min(
      this.initialReconnectDelay * Math.pow(this.backoffMultiplier, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    );

    logger.info(`Reconnecting WebSocket (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`, {
      delay,
      nextAttemptAt: new Date(Date.now() + delay).toISOString(),
    });

    this.emit('reconnecting', this.reconnectAttempts, this.maxReconnectAttempts);

    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch(error => {
        logger.error('Reconnection attempt failed', error as Error);
        this.emit('error', error);
      });
    }, delay);
  }

  /**
   * Start heartbeat mechanism to detect stale connections
   */
  private startHeartbeat() {
    this.stopHeartbeat();
    this.lastHeartbeatTime = Date.now();

    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        // Check if we've missed too many heartbeats
        if (this.lastHeartbeatTime && Date.now() - this.lastHeartbeatTime > this.heartbeatInterval * this.MAX_MISSED_HEARTBEATS) {
          logger.warn('WebSocket connection appears stale, reconnecting');
          this.socket.close();
          return;
        }

        // Send ping
        this.send({
          type: 'ping',
          payload: {},
          timestamp: new Date().toISOString(),
          id: crypto.randomUUID(),
        });

        this.missedHeartbeats++;

        if (this.missedHeartbeats >= this.MAX_MISSED_HEARTBEATS) {
          logger.warn('Too many missed heartbeats, closing connection');
          this.socket.close();
        }
      }
    }, this.heartbeatInterval);

    logger.debug('Heartbeat started', { interval: this.heartbeatInterval });
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
      logger.debug('Heartbeat stopped');
    }
  }

  public subscribe(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.unsubscribe(handler);
  }

  public unsubscribe(handler: MessageHandler): void {
    this.messageHandlers.delete(handler);
  }

  public on(event: EventType, handler: Function): () => void {
    const handlers = this.eventHandlers.get(event) || new Set();
    handlers.add(handler);
    this.eventHandlers.set(event, handlers);

    return () => this.off(event, handler);
  }

  public off(event: EventType, handler: Function): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  private emit(event: EventType, ...args: any[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(...args);
        } catch (error) {
          logger.error(`Error in ${event} handler`, error as Error);
        }
      });
    }
  }

  // Send a scan update subscription
  public subscribeToScan(scanId: string): void {
    this.send({
      type: 'subscribe',
      channel: `scan_${scanId}`,
      action: 'subscribe'
    });
  }

  // Unsubscribe from scan updates
  public unsubscribeFromScan(scanId: string): void {
    this.send({
      type: 'unsubscribe',
      channel: `scan_${scanId}`,
      action: 'unsubscribe'
    });
  }

  // Get current connection status
  public getConnectionStatus(): 'connected' | 'disconnected' | 'connecting' {
    if (!this.socket) return 'disconnected';
    switch (this.socket.readyState) {
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CONNECTING:
        return 'connecting';
      default:
        return 'disconnected';
    }
  }

  /**
   * Send a message through the WebSocket with queuing support
   */
  public send(data: WebSocketMessage | any): void {
    const message: WebSocketMessage = typeof data === 'object' && data !== null && 'type' in data && 'payload' in data
      ? data as WebSocketMessage
      : {
        type: 'ping',
        payload: data || {},
        timestamp: new Date().toISOString(),
        id: crypto.randomUUID(),
      } as WebSocketMessage;

    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      try {
        const messageStr = JSON.stringify(message);
        this.socket.send(messageStr);

        logger.debug('WebSocket message sent', { type: message.type });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        logger.error('Failed to send WebSocket message', error as Error);
        this.emit('error', new Error(`Failed to send message: ${errorMessage}`));

        // Queue message for retry
        this.queueMessage(message);
      }
    } else {
      logger.debug('WebSocket not connected, queuing message', { type: message.type });
      this.queueMessage(message);

      // Try to reconnect if not already reconnecting
      if (this.connectionState === 'disconnected' && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.connect().catch((error) => {
          logger.error('Failed to reconnect for queued message', error);
        });
      }
    }
  }

  /**
   * Queue a message for later delivery
   */
  private queueMessage(message: WebSocketMessage): void {
    if (this.messageQueue.length >= this.maxQueueSize) {
      // Remove oldest message
      const removed = this.messageQueue.shift();
      logger.warn('Message queue full, removing oldest message', { removedType: removed?.message.type });
    }

    this.messageQueue.push({
      message,
      timestamp: new Date().toISOString(),
      retries: 0,
      priority: 'normal',
    });

    logger.debug('Message queued', {
      type: message.type,
      queueSize: this.messageQueue.length,
    });
  }

  /**
   * Process queued messages after reconnection
   */
  private async processMessageQueue(): Promise<void> {
    if (this.isProcessingQueue || this.messageQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;
    logger.info('Processing message queue', { count: this.messageQueue.length });

    const queue = [...this.messageQueue];
    this.messageQueue = [];

    for (const queuedMessage of queue) {
      if (this.socket?.readyState === WebSocket.OPEN) {
        try {
          const messageStr = JSON.stringify(queuedMessage.message);
          this.socket.send(messageStr);
          logger.debug('Queued message sent', { type: queuedMessage.message.type });
        } catch (error) {
          logger.error('Failed to send queued message', error as Error);
          // Re-queue if retries available
          if (queuedMessage.retries < 3) {
            queuedMessage.retries++;
            this.messageQueue.push(queuedMessage);
          }
        }
      } else {
        // Connection lost, re-queue all remaining messages
        this.messageQueue.push(queuedMessage);
        break;
      }
    }

    this.isProcessingQueue = false;
    logger.info('Message queue processed', {
      sent: queue.length - this.messageQueue.length,
      remaining: this.messageQueue.length,
    });
  }

  /**
   * Disconnect the WebSocket
   */
  public disconnect(): void {
    // Clear reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.socket) {
      this.stopHeartbeat();
      this.connectionState = 'disconnecting';
      this.socket.close(1000, 'User disconnected');
      this.socket = null;
      this.isAuthenticated = false;
      this.connectionState = 'disconnected';
      logger.info('WebSocket disconnected by user');
      this.emit('disconnected', new CloseEvent('close', { code: 1000, reason: 'User disconnected' }));
    }
  }

  /**
   * Get current connection state
   */
  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.connectionState === 'connected' && this.socket?.readyState === WebSocket.OPEN;
  }

  /**
   * Check if authenticated
   */
  public isAuth(): boolean {
    return this.isAuthenticated;
  }

  /**
   * Get queue size
   */
  public getQueueSize(): number {
    return this.messageQueue.length;
  }

  /**
   * Clear message queue
   */
  public clearQueue(): void {
    const count = this.messageQueue.length;
    this.messageQueue = [];
    logger.info('Message queue cleared', { count });
  }
}

// Export the WebSocketService class and create a default instance
export const webSocketService = WebSocketService.getInstance({
  maxReconnectAttempts: 5,
  initialReconnectDelay: 1000,
  maxReconnectDelay: 30000,
  backoffMultiplier: 2,
  heartbeatInterval: 30000,
  messageQueueSize: 100,
  debug: import.meta.env.DEV
});

export default webSocketService;
