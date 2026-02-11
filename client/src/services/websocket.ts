/**
 * Enhanced WebSocket Service
 * Provides reliable WebSocket connections with exponential backoff,
 * message queuing, and automatic reconnection
 */

import { getAccessToken } from '../lib/auth/getAccessToken';
import { API_CONFIG } from '../lib/config/urls';

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
import {
  isValidMessage,
  createMessage,
  type WebSocketMessage,
  type ConnectionState,
  type MessageHandler,
  type EventType,
  type QueuedMessage,
  type ErrorMessage
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
  private missedHeartbeats = 0;
  private readonly MAX_MISSED_HEARTBEATS = 3;

  // Connection state
  private connectionState: ConnectionState = 'disconnected';
  private isAuthenticated = false;
  private isProcessingQueue = false;
  private config: {
    connectionTimeout: number;
    maxReconnectAttempts: number;
    initialReconnectDelay: number;
    maxReconnectDelay: number;
    backoffMultiplier: number;
    heartbeatInterval: number;
    messageQueueSize: number;
  };

  constructor(options: WebSocketOptions = {}) {
    // Initialize config with defaults or provided options
    this.config = {
      connectionTimeout: 10000, // 10 seconds
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      initialReconnectDelay: options.initialReconnectDelay || 1000,
      maxReconnectDelay: options.maxReconnectDelay || 30000,
      backoffMultiplier: options.backoffMultiplier || 2,
      heartbeatInterval: options.heartbeatInterval || 30000,
      messageQueueSize: options.messageQueueSize || 100
    };

    // Set individual properties for backward compatibility
    this.maxReconnectAttempts = this.config.maxReconnectAttempts;
    this.initialReconnectDelay = this.config.initialReconnectDelay;
    this.maxReconnectDelay = this.config.maxReconnectDelay;
    this.backoffMultiplier = this.config.backoffMultiplier;
    this.heartbeatInterval = this.config.heartbeatInterval;
    this.maxQueueSize = this.config.messageQueueSize;

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

  public async connect(token?: string): Promise<boolean> {
    // Prevent multiple connection attempts
    if (this.socket && (this.socket.readyState === WebSocket.CONNECTING)) {
      logger.debug('WebSocket connection already in progress');
      return false;
    }

    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      logger.debug('WebSocket already connected');
      return true;
    }

    try {
      this.setState('connecting');
      
      // Clean up any existing connection
      this.cleanup();

      // Get auth token if not provided
      const authToken = token || (await getAuthToken());
      if (!authToken) {
        logger.error('No authentication token available for WebSocket connection');
        this.handleError(new Error('Authentication required'));
        return false;
      }

      // Construct WebSocket URL with token
      const wsUrl = this.getWebSocketUrl(authToken);
      
      logger.info('Attempting WebSocket connection...', { 
        url: this.sanitizeUrl(wsUrl),
        attempt: this.reconnectAttempts + 1
      });

      // Create new WebSocket connection with timeout
      this.socket = new WebSocket(wsUrl);
      this.setupEventHandlers();

      // Set up connection timeout with longer duration
      return await new Promise<boolean>((resolve) => {
        const connectionTimeout = setTimeout(() => {
          logger.error('WebSocket connection timeout after 10 seconds');
          this.emit('error', new Error('Connection timeout'));
          this.scheduleReconnect();
          resolve(false);
        }, 10000); // 10 second timeout

        const onOpen = () => {
          clearTimeout(connectionTimeout);
          logger.info('WebSocket connection established successfully');
          this.setState('connected');
          this.reconnectAttempts = 0;
          this.setupHeartbeat();
          this.flushMessageQueue();
          this.emit('connected');
          resolve(true);
        };

        const onError = (event: Event) => {
          clearTimeout(connectionTimeout);
          const error = this.createErrorFromEvent(event);
          logger.error('WebSocket connection failed:', error);
          this.handleError(error);
          resolve(false);
        };

        this.socket?.addEventListener('open', onOpen, { once: true });
        this.socket?.addEventListener('error', onError, { once: true });
      });
    } catch (error) {
      const errorObj = this.createErrorFromUnknown(error);
      logger.error('Failed to establish WebSocket connection:', errorObj);
      this.handleError(errorObj);
      this.scheduleReconnect();
      return false;
    }
  }

  private getWebSocketUrl(token: string): string {
    try {
      // Use centralized WebSocket URL configuration
      const baseUrl = API_CONFIG.WS_URL;
      
      // Parse base URL
      const url = new URL(baseUrl);
      
      // Handle localhost correctly
      if (url.hostname === 'localhost') {
        url.hostname = '127.0.0.1';
      }
      
      // Add token as query parameter
      url.searchParams.set('token', token);
      
      // Add timestamp to prevent caching
      url.searchParams.set('t', Date.now().toString());
      
      const finalUrl = url.toString();
      logger.debug('Constructed WebSocket URL', { 
        url: this.sanitizeUrl(finalUrl)
      });
      
      return finalUrl;
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

  private setupEventHandlers(): void {
    if (!this.socket) {
      return;
    }

    this.socket.onopen = this.handleOpen;
    this.socket.onmessage = this.handleMessage;
    this.socket.onclose = this.handleClose;
    this.socket.onerror = (event: Event) => {
      this.handleError(this.createErrorFromEvent(event));
    };
  }

  private handleOpen = (): void => {
    this.reconnectAttempts = 0;
    this.setState('connected');
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
      
      // Emit message event
      this.emit('message', message);
      
      // Call registered message handlers
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
      
      // Reset missed heartbeats for any valid message
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
    this.setState('disconnected');
    this.emit('disconnected', event);

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
  };

  private handleError = (error: Error): void => {
    // Log the error with context
    logger.error('WebSocket error occurred', error);
    
    // Emit the error with the error object
    this.emit('error', error);
    
    // If we're not already reconnecting, try to reconnect
    if (this.connectionState !== 'connecting' && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  };

  private createErrorFromEvent(event: Event): Error {
    if (event instanceof ErrorEvent) {
      return new Error(event.message);
    }
    return new Error('WebSocket error occurred');
  }

  private createErrorFromUnknown(error: unknown): Error {
    if (error instanceof Error) {
      return error;
    }
    if (typeof error === 'string') {
      return new Error(error);
    }
    return new Error('Unknown error occurred');
  }

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
    this.setState('reconnecting');
    this.emit('reconnecting', { attempt: this.reconnectAttempts, delay });

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

    // Close the socket if it exists
    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onmessage = null;
      this.socket.onclose = null;
      this.socket.onerror = null;
      
      if (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING) {
        this.socket.close();
      }
      this.socket = null;
    }
    
    this.isAuthenticated = false;
    this.missedHeartbeats = 0;
    this.isProcessingQueue = false;
  }

  private setupHeartbeat(): void {
    this.startHeartbeat();
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.missedHeartbeats >= this.MAX_MISSED_HEARTBEATS) {
        logger.warn('Max missed heartbeats reached, reconnecting...');
        this.reconnect();
        return;
      }

      if (this.socket?.readyState === WebSocket.OPEN) {
        this.sendPing();
        this.missedHeartbeats++;
      }
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    this.missedHeartbeats = 0;
  }

  private sendPing(): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      const pingMessage: WebSocketMessage = {
        type: 'ping',
        timestamp: new Date().toISOString(),
        id: `ping-${Date.now()}`,
        payload: {
          serverTime: new Date().toISOString(),
          clientTime: new Date().toISOString()
        }
      };
      this.socket.send(JSON.stringify(pingMessage));
    }
  }

  public sendMessage(message: unknown): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.queueMessage(message);
      return false;
    }

    try {
      const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
      this.socket.send(messageStr);
      return true;
    } catch (error) {
      logger.error('Error sending WebSocket message:', this.createErrorFromUnknown(error));
      this.queueMessage(message);
      return false;
    }
  }

  private queueMessage(message: unknown): void {
    if (this.messageQueue.length >= this.maxQueueSize) {
      logger.warn('Message queue full, dropping oldest message');
      this.messageQueue.shift();
    }

    // Convert unknown message to proper WebSocketMessage format
    let wsMessage: WebSocketMessage;
    if (typeof message === 'string') {
      try {
        const parsed = JSON.parse(message);
        wsMessage = isValidMessage(parsed) ? parsed : createMessage('error', { 
          code: 'INVALID_MESSAGE', 
          message: 'Invalid message format', 
          recoverable: false 
        }) as ErrorMessage;
      } catch {
        wsMessage = createMessage('error', { 
          code: 'INVALID_MESSAGE', 
          message: 'Invalid JSON message', 
          recoverable: false 
        }) as ErrorMessage;
      }
    } else if (isValidMessage(message)) {
      wsMessage = message as WebSocketMessage;
    } else {
      wsMessage = createMessage('error', { 
        code: 'INVALID_MESSAGE', 
        message: 'Message does not conform to WebSocketMessage interface', 
        recoverable: false 
      }) as ErrorMessage;
    }

    this.messageQueue.push({
      message: wsMessage,
      timestamp: new Date().toISOString(),
      retries: 0,
      priority: 'normal' as const
    });
  }

  private processMessageQueue(): void {
    this.flushMessageQueue();
  }

  private flushMessageQueue(): void {
    if (this.isProcessingQueue || !this.isConnected()) {
      return;
    }

    this.isProcessingQueue = true;

    try {
      while (this.messageQueue.length > 0 && this.isConnected()) {
        const queuedMessage = this.messageQueue[0];
        if (this.sendMessage(queuedMessage.message)) {
          this.messageQueue.shift();
        } else {
          break;
        }
      }
    } catch (error) {
      logger.error('Error flushing message queue:', this.createErrorFromUnknown(error));
    } finally {
      this.isProcessingQueue = false;
    }
  }

  public disconnect(): void {
    this.isShuttingDown = true;
    this.stopHeartbeat();

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.socket) {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.close(1000, 'User disconnected');
      }
      this.socket = null;
    }

    this.setState('disconnected');
  }

  public async reconnect(): Promise<boolean> {
    if (this.socket?.readyState === WebSocket.CONNECTING) {
      return false;
    }
    
    this.cleanup();
    this.setState('reconnecting');
    this.emit('reconnecting', { manual: true });
    
    try {
      const connected = await this.connect();
      if (!connected) {
        throw new Error('Failed to establish WebSocket connection');
      }
      return true;
    } catch (error) {
      const errorObj = this.createErrorFromUnknown(error);
      logger.error('Reconnection failed:', errorObj);
      this.emit('error', errorObj);
      this.scheduleReconnect();
      return false;
    }
  }

  private setState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      logger.debug(`WebSocket state changed to: ${state}`);
    }
  }

  private emit(event: EventType, data?: unknown): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Error in ${event} event handler: ${errorMessage}`);
          if (error instanceof Error && error.stack) {
            logger.debug('Error stack:', error.stack);
          }
        }
      });
    }
  }

  /**
   * Add event listener for WebSocket events
   */
  public addEventListener(event: EventType, handler: (...args: unknown[]) => void): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.add(handler);
    }
  }

  /**
   * Remove event listener for WebSocket events
   */
  public removeEventListener(event: EventType, handler: (...args: unknown[]) => void): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * On method (alias for addEventListener)
   */
  public on(event: EventType, handler: (...args: unknown[]) => void): () => void {
    this.addEventListener(event, handler);
    return () => this.removeEventListener(event, handler);
  }

  /**
   * Off method (alias for removeEventListener)
   */
  public off(event: EventType, handler: (...args: unknown[]) => void): void {
    this.removeEventListener(event, handler);
  }

  /**
   * Subscribe to WebSocket messages (convenience method)
   */
  public subscribe(handler: MessageHandler): () => void {
    this.addMessageHandler(handler);
    return () => this.removeMessageHandler(handler);
  }

  /**
   * Add message handler for WebSocket messages
   */
  public addMessageHandler(handler: MessageHandler): void {
    this.messageHandlers.add(handler);
  }

  /**
   * Remove message handler for WebSocket messages
   */
  public removeMessageHandler(handler: MessageHandler): void {
    this.messageHandlers.delete(handler);
  }

  /**
   * On message method (convenience for addMessageHandler)
   */
  public onMessage(handler: MessageHandler): () => void {
    this.addMessageHandler(handler);
    return () => this.removeMessageHandler(handler);
  }

  /**
   * Off message method (convenience for removeMessageHandler)
   */
  public offMessage(handler: MessageHandler): void {
    this.removeMessageHandler(handler);
  }

  /**
   * Send a message through WebSocket
   */
  public send(message: unknown): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.queueMessage(message);
      return false;
    }

    try {
      const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
      this.socket.send(messageStr);
      return true;
    } catch (error) {
      logger.error('Error sending WebSocket message:', this.createErrorFromUnknown(error));
      this.queueMessage(message);
      return false;
    }
  }

  /**
   * Subscribe to scan updates
   */
  public subscribeToScan(scanId: string): () => void {
    const message = {
      type: 'subscribe',
      payload: { channel: `scan:${scanId}` },
      timestamp: new Date().toISOString(),
      id: `subscribe-scan-${scanId}-${Date.now()}`
    };
    this.send(message);
    
    return () => {
      this.unsubscribeFromScan(scanId);
    };
  }

  /**
   * Unsubscribe from scan updates
   */
  public unsubscribeFromScan(scanId: string): void {
    const message = {
      type: 'unsubscribe',
      payload: { channel: `scan:${scanId}` },
      timestamp: new Date().toISOString(),
      id: `unsubscribe-scan-${scanId}-${Date.now()}`
    };
    this.send(message);
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN && this.connectionState === 'connected';
  }

  /**
   * Check if authenticated
   */
  public get isAuth(): boolean {
    return this.isAuthenticated;
  }

  /**
   * Get current connection state
   */
  public getState(): ConnectionState {
    return this.connectionState;
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

  /**
   * Get reconnection attempts
   */
  public getReconnectAttempts(): number {
    return this.reconnectAttempts;
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