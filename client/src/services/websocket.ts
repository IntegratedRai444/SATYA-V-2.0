import { getAuthToken } from './auth';

export type MessageType = 
  | 'scan_update'
  | 'notification'
  | 'system_alert'
  | 'heartbeat'
  | 'auth_response'
  | 'error';

export type ScanStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface BaseMessage {
  type: MessageType;
  timestamp: string;
  [key: string]: any;
}

export interface ScanUpdateMessage extends BaseMessage {
  type: 'scan_update';
  scanId: string;
  status: ScanStatus;
  progress?: number;
  fileName?: string;
  error?: string;
  data?: any;
}

export interface NotificationMessage extends BaseMessage {
  type: 'notification';
  id: string;
  title: string;
  message: string;
  severity?: 'info' | 'success' | 'warning' | 'error';
  read?: boolean;
}

export interface SystemAlertMessage extends BaseMessage {
  type: 'system_alert';
  id: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface HeartbeatMessage extends BaseMessage {
  type: 'heartbeat';
}

export interface AuthResponseMessage extends BaseMessage {
  type: 'auth_response';
  authenticated: boolean;
  userId?: string;
  error?: string;
}

export interface ErrorMessage extends BaseMessage {
  type: 'error';
  code: string;
  message: string;
  details?: any;
}

export type WebSocketMessage = 
  | ScanUpdateMessage
  | NotificationMessage
  | SystemAlertMessage
  | HeartbeatMessage
  | AuthResponseMessage
  | ErrorMessage;

type MessageHandler = (message: WebSocketMessage) => void;
type EventType = 'connected' | 'disconnected' | 'error' | 'reconnecting' | 'message' | 'scan_update' | 'notification' | 'system_alert';

interface WebSocketHandlers {
  onConnected?: () => void;
  onDisconnected?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onReconnecting?: (attempt: number, maxAttempts: number) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onScanUpdate?: (message: ScanUpdateMessage) => void;
  onNotification?: (message: NotificationMessage) => void;
  onSystemAlert?: (message: SystemAlertMessage) => void;
}

interface WebSocketOptions {
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  debug?: boolean;
}

class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private eventHandlers: Map<EventType, Set<Function>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private debug: boolean;
  private isAuthenticated = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private readonly HEARTBEAT_INTERVAL = 30000; // 30 seconds

  constructor(options: WebSocketOptions = {}) {
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
    this.reconnectDelay = options.reconnectDelay || 3000;
    this.debug = options.debug || process.env.NODE_ENV === 'development';
    
    // Initialize event handlers
    (['connected', 'disconnected', 'error', 'reconnecting', 'message'] as EventType[]).forEach(event => {
      this.eventHandlers.set(event, new Set());
    });
  }

  public static getInstance(options?: WebSocketOptions): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(options);
    }
    return WebSocketService.instance;
  }

  public async connect(token?: string): Promise<void> {
    if (this.socket) {
      this.disconnect();
    }

    const authToken = token || (await getAuthToken());
    if (!authToken) {
      this.emit('error', new Error('No authentication token available'));
      return;
    }

    try {
      const wsUrl = this.getWebSocketUrl(authToken);
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);
    } catch (error) {
      this.emit('error', error);
      this.handleReconnect();
    }
  }

  private getWebSocketUrl(token: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const host = process.env.REACT_APP_WS_HOST || window.location.host;
    const path = process.env.REACT_APP_WS_PATH || '/ws';
    return `${protocol}${host}${path}?token=${encodeURIComponent(token)}`;
  }

  private handleOpen() {
    this.reconnectAttempts = 0;
    this.isAuthenticated = true;
    this.startHeartbeat();
    this.emit('connected');
    this.log('WebSocket connected');
  }

  private handleMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data) as WebSocketMessage;
      
      // Handle heartbeat response
      if (data.type === 'heartbeat' && data.timestamp) {
        this.send({ type: 'heartbeat', timestamp: new Date().toISOString() });
        return;
      }
      
      // Handle authentication response
      if (data.type === 'auth_response') {
        this.isAuthenticated = data.authenticated;
        if (!data.authenticated) {
          this.emit('error', new Error(data.error || 'Authentication failed'));
          this.disconnect();
        } else {
          this.emit('connected');
        }
        return;
      }

      // Emit specific event types
      this.emit('message', data);
      
      // Emit type-specific events
      switch (data.type) {
        case 'scan_update':
          this.emit('scan_update', data as ScanUpdateMessage);
          break;
        case 'notification':
          this.emit('notification', data as NotificationMessage);
          break;
        case 'system_alert':
          this.emit('system_alert', data as SystemAlertMessage);
          break;
      }
      
      // Notify all message handlers
      this.messageHandlers.forEach(handler => handler(data));
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      this.emit('error', error instanceof Error ? error : new Error('Error parsing WebSocket message'));
    }
  }

  private handleClose(event: CloseEvent) {
    this.stopHeartbeat();
    this.isAuthenticated = false;
    this.emit('disconnected', event);
    this.log('WebSocket disconnected', event);
    
    // Only attempt to reconnect if the close was unexpected
    if (event.code !== 1000) { // 1000 is a normal closure
      this.handleReconnect();
    }
  }

  private handleError(error: Event) {
    this.emit('error', error);
    this.log('WebSocket error:', error);
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1); // Exponential backoff
      
      this.emit('reconnecting', {
        attempt: this.reconnectAttempts,
        maxAttempts: this.maxReconnectAttempts,
        nextAttemptIn: delay
      });
      
      this.log(`Reconnecting in ${delay}ms... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      this.reconnectTimeout = setTimeout(() => {
        this.connect().catch(error => {
          this.emit('error', error);
        });
      }, delay);
    } else {
      this.emit('error', new Error('Max reconnection attempts reached'));
      this.log('Max reconnection attempts reached');
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, this.HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
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
          console.error(`Error in ${event} handler:`, error);
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

  // Send a message through the WebSocket
  public send<T = any>(data: T): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      try {
        const message = typeof data === 'string' ? data : JSON.stringify(data);
        this.socket.send(message);
        
        if (this.debug) {
          console.debug('WebSocket sent:', data);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        this.emit('error', new Error(`Failed to send message: ${errorMessage}`));
      }
    } else {
      this.emit('error', new Error('WebSocket is not connected'));
      // Try to reconnect if not already reconnecting
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.connect().catch(() => {
          this.emit('error', new Error('Failed to reconnect'));
        });
      }
    }
  }

  // Disconnect the WebSocket
  public disconnect(): void {
    if (this.socket) {
      this.stopHeartbeat();
      this.socket.close(1000, 'User disconnected');
      this.socket = null;
      this.isAuthenticated = false;
      this.emit('disconnected');
    }
  }

  private log(...args: any[]): void {
    if (this.debug) {
      console.log('[WebSocket]', ...args);
    }
  }
}

// Export the WebSocketService class and create a default instance
export const webSocketService = WebSocketService.getInstance({
  maxReconnectAttempts: 5,
  reconnectDelay: 3000,
  debug: process.env.NODE_ENV === 'development'
});

// Export types
export type {
  WebSocketMessage,
  ScanUpdateMessage,
  NotificationMessage,
  SystemAlertMessage,
  HeartbeatMessage,
  AuthResponseMessage,
  ErrorMessage,
  MessageType,
  ScanStatus,
  MessageHandler,
  EventType
};
