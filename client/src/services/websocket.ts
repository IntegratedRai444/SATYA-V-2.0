/**
 * Enhanced WebSocket Service
 * Provides reliable WebSocket connections with exponential backoff,
 * message queuing, and automatic reconnection
 */

import { getAccessToken } from '../lib/auth/getAccessToken';

// Get JWT token for WebSocket authentication
const getAuthToken = async (): Promise<string | null> => {
  try {
    const token = await getAccessToken();
    if (import.meta.env.DEV) {
      console.log("WebSocket auth token:", token ? "Bearer [REDACTED]" : "null");
    }
    return token;
  } catch (error) {
    console.error('Error getting auth token for WebSocket:', error);
    return null;
  }
};
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
  private eventHandlers: Map<EventType, Set<(...args: unknown[]) => void>> = new Map();
  private isShuttingDown = false;
  // Reconnection with exponential backoff and jitter
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private initialReconnectDelay: number;
  private maxReconnectDelay: number;
  private backoffMultiplier: number;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  // Message queue for offline messages
  private messageQueue: QueuedMessage[] = [];
  private maxQueueSize: number;

  // Heartbeat mechanism
  private heartbeatInterval: number;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
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
    if (this.isShuttingDown) {
      logger.debug('WebSocket is shutting down, ignoring connect request');
      return;
    }

    if (this.socket && this.connectionState === 'connected') {
      logger.debug('WebSocket already connected');
      return;
    }

    // Clear any existing connection
    this.disconnect();

    try {
      this.connectionState = 'connecting';
      const authToken = token || (await getAuthToken());
      
      if (!authToken) {
        throw new Error('No authentication token available');
      }

      const wsUrl = this.getWebSocketUrl(authToken);
      logger.debug('Connecting to WebSocket', { url: this.sanitizeUrl(wsUrl) });

      this.socket = new WebSocket(wsUrl);
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);

      // Set up heartbeat after connection is established
      this.startHeartbeat();
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect to WebSocket';
      logger.error('WebSocket connection failed: ' + errorMessage);
      this.emit('error', error instanceof Error ? error : new Error(errorMessage));
      this.connectionState = 'disconnected';
      
      // Schedule reconnection if not shutting down
      if (!this.isShuttingDown) {
        this.scheduleReconnect();
      }
    }
  }

  private getWebSocketUrl(token: string): string {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const configured = import.meta.env.VITE_WS_URL;

      // VITE_WS_URL can be either:
      // - host only: "localhost:5001"
      // - full URL: "ws://localhost:5001/api/v2/dashboard/ws"
      // - full URL without protocol: "localhost:5001/api/v2/dashboard/ws"
      if (configured) {
        const hasProtocol = /^wss?:\/\//i.test(configured);
        const baseUrl = hasProtocol ? configured : `${protocol}//${configured}`;
        const url = new URL(baseUrl);

        // If a path is already provided, use it; otherwise default to the dashboard ws endpoint
        const path = url.pathname && url.pathname !== '/' ? url.pathname : '/api/v2/dashboard/ws';
        url.pathname = path;
        url.searchParams.set('token', token);
        return url.toString();
      }

      return `${protocol}//localhost:5001/api/v2/dashboard/ws?token=${encodeURIComponent(token)}`;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Failed to construct WebSocket URL: ' + errorMessage);
      throw new Error('Failed to construct WebSocket URL: ' + errorMessage);
    }
  }

  private sanitizeUrl(url: string): string {
    // Remove sensitive information from URL for logging
    return url.replace(/token=[^&]+/, 'token=***');
  }

  private handleOpen = (): void => {
    this.reconnectAttempts = 0;
    this.updateConnectionState('connected');
    this.isAuthenticated = true;
    this.missedHeartbeats = 0;

    logger.info('WebSocket connected successfully');
    this.startHeartbeat();
    this.emit('connected');

    // Process queued messages
    this.processMessageQueue();
  };

  private handleMessage = (event: MessageEvent): void => {
    try {
      if (!event.data) {
        logger.warn('Received empty WebSocket message');
        return;
      }

      const message = JSON.parse(event.data) as WebSocketMessage;
      this.messageHandlers.forEach(handler => {
        try {
          handler(message);
        } catch (handlerError) {
          const errorMessage = `Error in message handler: ${
            handlerError instanceof Error ? handlerError.message : String(handlerError)
          }${message?.type ? ` (message type: ${message.type})` : ''}`;
          logger.error(errorMessage);
          
          // Emit error event instead of swallowing
          this.emit('error', new Error(errorMessage));
        }
      });
      
      // Update last heartbeat time for any message (not just heartbeats)
      this.lastHeartbeatTime = Date.now();
      this.missedHeartbeats = 0;
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error handling WebSocket message';
      logger.error(`${errorMessage}. Data: ${JSON.stringify(event.data).substring(0, 200)}`);
      if (error instanceof Error && error.stack) {
        logger.debug('Stack trace: ' + error.stack);
      }
      
      this.emit('error', new Error(errorMessage));
    }
  };

  private handleClose = (event: CloseEvent): void => {
    logger.debug('WebSocket connection closed', { 
      code: event.code, 
      reason: event.reason,
      wasClean: event.wasClean,
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.maxReconnectAttempts
    });

    this.cleanup();
    this.updateConnectionState('disconnected');

    // Don't attempt to reconnect if we're shutting down or reached max attempts
    if (this.isShuttingDown || this.reconnectAttempts >= this.maxReconnectAttempts) {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        const error = new Error('Max reconnection attempts reached');
        logger.error(error.message);
        this.emit('error', error);
      }
      return;
    }

    // Only attempt to reconnect if the close was not clean
    if (!event.wasClean) {
      this.scheduleReconnect();
    }
  }

  private handleError = (event: Event | ErrorEvent): void => {
    const errorMessage = event instanceof ErrorEvent ? event.message : 'WebSocket error occurred';
    const error = new Error(errorMessage);
    
    // Prepare error context
    const errorContext: Record<string, unknown> = {
      name: 'WebSocketError',
      ...(event instanceof ErrorEvent && { originalError: event.error }),
      ...('type' in event && { type: (event as { type: string }).type }),
      ...('code' in event && { code: (event as { code: number }).code })
    };
    
    // Log the error with context
    logger.error('WebSocket error occurred', error, errorContext);
    
    // Emit the error with the error object
    this.emit('error', error);
    
    // If we're not already reconnecting, try to reconnect
    if (this.connectionState !== 'connecting' && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  };

  private scheduleReconnect(): void {
    if (this.reconnectTimeout || this.isShuttingDown) {
      return; // Already scheduled or shutting down
    }

    // Calculate delay with exponential backoff and jitter
    const baseDelay = Math.min(
      this.initialReconnectDelay * Math.pow(this.backoffMultiplier, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    
    // Add jitter (randomness) to prevent thundering herd problem
    const jitter = Math.random() * 0.5 * baseDelay; // up to 50% jitter
    const delay = Math.floor(baseDelay + jitter);

    this.reconnectAttempts++;
    this.updateConnectionState('reconnecting');

    logger.debug(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      if (this.isShuttingDown) {
        logger.debug('Skipping reconnection - service is shutting down');
        return;
      }
      
      this.connect().catch(error => {
        logger.error('Reconnection attempt failed', error, { 
          attempt: this.reconnectAttempts,
          maxAttempts: this.maxReconnectAttempts 
        });
        this.emit('error', error);
      });
    }, delay);
  }

  private cleanup(): void {
    // Clear all timeouts and intervals
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Clean up WebSocket
    if (this.socket) {
      try {
        // Remove all event listeners
        this.socket.onopen = null;
        this.socket.onmessage = null;
        this.socket.onclose = null;
        this.socket.onerror = null;
        
        // Only try to close if the connection is open or connecting
        if (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING) {
          this.socket.close(1000, 'Connection closed by client');
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error : new Error('Unknown error during cleanup');
        logger.error('Error during WebSocket cleanup', errorMessage);
        this.emit('error', errorMessage as Error);
      } finally {
        this.socket = null;
      }
    }
  }

  /**
   * Completely shut down the WebSocket service
   */
  public shutdown(): void {
    this.isShuttingDown = true;
    this.cleanup();
    this.messageHandlers.clear();
    this.eventHandlers.clear();
    this.messageQueue = [];
    this.reconnectAttempts = 0;
    this.connectionState = 'disconnected';
  }

  private startHeartbeat(): void {
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

        this.sendPing();

        this.missedHeartbeats++;

        if (this.missedHeartbeats >= this.MAX_MISSED_HEARTBEATS) {
          logger.warn('Too many missed heartbeats, closing connection');
          this.socket.close();
        }
      }
    }, this.heartbeatInterval);

    logger.debug('Heartbeat started', { interval: this.heartbeatInterval });
  }

  private sendPing(): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      const pingMessage: WebSocketMessage = {
        type: 'ping',
        payload: { serverTime: new Date().toISOString() },
        timestamp: new Date().toISOString(),
        id: crypto.randomUUID()
      };
      this.socket.send(JSON.stringify(pingMessage));
    }
  }

  private stopHeartbeat(): void {
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

  public on(event: EventType, handler: (...args: unknown[]) => void): () => void {
    const handlers = this.eventHandlers.get(event) || new Set();
    handlers.add(handler);
    this.eventHandlers.set(event, handlers);
    return () => this.off(event, handler);
  }

  public off(event: EventType, handler: (...args: unknown[]) => void): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  private emit(event: EventType, ...args: unknown[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(...args);
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Error in ${event} handler: ${errorMessage}`);
        }
      });
    }
  }

  private isProcessingQueue = false;

  /**
   * Send a scan update subscription
   */
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
  public send(data: WebSocketMessage | Record<string, unknown>): void {
    const message: WebSocketMessage = typeof data === 'object' && data !== null && 'type' in data && 'payload' in data
      ? data as WebSocketMessage
      : {
        type: 'ping',
        payload: { 
          serverTime: new Date().toISOString(),
          clientTime: new Date().toISOString()
        },
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
          this.emit('error', error as Error);
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
          this.emit('error', error as Error);
          
          // Re-queue if retries available
          if (queuedMessage.retries < 3) {
            queuedMessage.retries++;
            this.messageQueue.push(queuedMessage);
          } else {
            // Max retries exceeded, remove from queue
            logger.error('Max retries exceeded for queued message');
            logger.error(`Message type: ${queuedMessage.message.type}`);
            logger.error(`Retries: ${queuedMessage.retries}`);
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

  private updateConnectionState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      
      // Emit specific events for connection state changes
      if (state === 'connected') {
        this.emit('connected');
      } else if (state === 'disconnected') {
        this.emit('disconnected');
      } else if (state === 'connecting') {
        // No matching EventType for 'connecting', use 'reconnecting' as fallback
        this.emit('reconnecting', this.reconnectAttempts, this.maxReconnectAttempts);
      } else if (state === 'disconnecting') {
        // No matching EventType for 'disconnecting', use 'disconnected' as fallback
        this.emit('disconnected');
      }
    }
  }

  /**
   * Check if connected
   */
  public get isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Check if authenticated
   */
  public get isAuth(): boolean {
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
