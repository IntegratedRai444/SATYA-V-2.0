/**
 * WebSocket Connection Manager
 * Manages WebSocket lifecycle with auth stability requirements
 * - Only connects after auth is stable
 * - Prevents connection storms
 * - Implements proper backoff with jitter
 */

import { getSession } from './authStore';
import { API_CONFIG } from './config/urls';

export interface WebSocketManagerState {
  isConnecting: boolean;
  isConnected: boolean;
  connectionAttempts: number;
  lastError: string | null;
}

class WebSocketManager {
  private static instance: WebSocketManager;
  private state: WebSocketManagerState;
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private maxReconnectAttempts = 3;
  private baseReconnectDelay = 1000; // 1 second
  private maxReconnectDelay = 30000; // 30 seconds

  private constructor() {
    this.state = {
      isConnecting: false,
      isConnected: false,
      connectionAttempts: 0,
      lastError: null,
    };
  }

  public static getInstance(): WebSocketManager {
    if (!WebSocketManager.instance) {
      WebSocketManager.instance = new WebSocketManager();
    }
    return WebSocketManager.instance;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout || this.state.isConnecting) {
      return;
    }

    // Exponential backoff with jitter
    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, this.state.connectionAttempts),
      this.maxReconnectDelay
    );
    
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.5 * delay;
    const finalDelay = Math.floor(delay + jitter);

    this.state.connectionAttempts += 1;
    this.state.isConnecting = true;

    console.log(`WebSocket reconnect attempt ${this.state.connectionAttempts} in ${finalDelay}ms (with jitter)`);

    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null;
      if (this.state.connectionAttempts <= this.maxReconnectAttempts) {
        this.connect();
      } else {
        console.error('Max WebSocket reconnection attempts reached. Giving up.');
        this.state.lastError = 'Max reconnection attempts reached';
      }
    }, finalDelay);
  }

  private connect(): void {
    const session = getSession();
    if (!session) {
      console.warn('Cannot connect WebSocket: user not authenticated');
      return;
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    this.state.isConnecting = true;
    const wsUrl = this.getWebSocketUrl(session?.access_token);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected successfully');
        this.state.isConnected = true;
        this.state.isConnecting = false;
        this.state.connectionAttempts = 0;
        this.state.lastError = null;
      };

      this.ws.onclose = (event: CloseEvent) => {
        console.log('WebSocket disconnected', { code: event.code, reason: event.reason });
        this.state.isConnected = false;
        this.state.isConnecting = false;
        this.ws = null;

        // Only reconnect if not a normal closure and we haven't exceeded max attempts
        if (event.code !== 1000 && this.state.connectionAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error: Event) => {
        console.error('WebSocket error:', error);
        this.state.lastError = 'Connection error';
        this.ws = null;
        this.state.isConnected = false;
        this.state.isConnecting = false;
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          // Handle message types here
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.state.lastError = 'Failed to create connection';
      this.state.isConnecting = false;
      this.ws = null;
    }
  }

  private getWebSocketUrl(token?: string): string {
    // Use centralized WebSocket URL configuration
    const baseUrl = API_CONFIG.WS_URL;
    const url = new URL(baseUrl);
    
    if (token) {
      url.searchParams.set('token', token);
    }
    
    return url.toString();
  }

  public connectIfAuthenticated(): void {
    const session = getSession();
    if (session && !this.state.isConnected && !this.state.isConnecting) {
      this.connect();
    }
  }

  public disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }

    this.state.isConnected = false;
    this.state.isConnecting = false;
    this.state.connectionAttempts = 0;
    this.state.lastError = null;
  }

  public getState(): WebSocketManagerState {
    return { ...this.state };
  }

  public sendMessage(message: Record<string, unknown>): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        const messageWithMeta = {
          ...message,
          timestamp: new Date().toISOString(),
          id: Math.random().toString(36).substr(2, 9),
        };
        this.ws.send(JSON.stringify(messageWithMeta));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    } else {
      console.warn('Cannot send message: WebSocket not connected');
      return false;
    }
  }
}

export default WebSocketManager;
