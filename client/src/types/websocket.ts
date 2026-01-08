/**
 * WebSocket Type Definitions
 * Comprehensive type definitions for WebSocket messages and events
 */

import type { AnalysisResult, Notification, DashboardStats } from './api';

// ============================================================================
// Base WebSocket Types
// ============================================================================

export type MessageType =
  | 'scan_update'
  | 'notification'
  | 'system_alert'
  | 'heartbeat'
  | 'auth_response'
  | 'dashboard_update'
  | 'error'
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe';

export type ConnectionState = 'connecting' | 'connected' | 'disconnecting' | 'disconnected' | 'reconnecting';

export type ScanStatus = 'queued' | 'processing' | 'completed' | 'failed';

// ============================================================================
// WebSocket Configuration
// ============================================================================

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnect: {
    enabled: boolean;
    maxAttempts: number;
    initialDelay: number;
    maxDelay: number;
    backoffMultiplier: number;
  };
  heartbeat: {
    enabled: boolean;
    interval: number;
    timeout: number;
  };
  messageQueue: {
    enabled: boolean;
    maxSize: number;
  };
  debug?: boolean;
}

// ============================================================================
// Base Message Interface
// ============================================================================

export interface BaseMessage<T = unknown> {
  type: MessageType;
  payload: T;
  timestamp: string;
  id: string;
}

// ============================================================================
// Specific Message Payloads
// ============================================================================

export interface ScanUpdatePayload {
  scanId: string;
  status: ScanStatus;
  progress: number;
  fileName?: string;
  fileType?: 'image' | 'video' | 'audio';
  error?: string;
  result?: AnalysisResult;
  estimatedTimeRemaining?: number;
}

export interface NotificationPayload extends Notification {
  priority?: 'low' | 'normal' | 'high' | 'urgent';
  category?: 'system' | 'analysis' | 'security' | 'update';
  dismissible?: boolean;
}

export interface SystemAlertPayload {
  id: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  timestamp: string;
  action?: {
    label: string;
    url?: string;
    callback?: string;
  };
  autoClose?: boolean;
  duration?: number;
}

export interface HeartbeatPayload {
  serverTime: string;
  clientTime?: string;
  latency?: number;
}

export interface AuthResponsePayload {
  authenticated: boolean;
  userId?: string;
  username?: string;
  error?: string;
  expiresAt?: string;
}

export interface DashboardUpdatePayload {
  updateType: 'stats' | 'analytics' | 'activity' | 'metrics' | 'all';
  data: Partial<DashboardStats>;
  timestamp: string;
}

export interface ErrorPayload {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  recoverable: boolean;
  retryAfter?: number;
}

export interface SubscribePayload {
  channel: string;
  filters?: Record<string, unknown>;
}

export interface UnsubscribePayload {
  channel: string;
}

// ============================================================================
// Typed Message Interfaces
// ============================================================================

export interface ScanUpdateMessage extends BaseMessage<ScanUpdatePayload> {
  type: 'scan_update';
}

export interface NotificationMessage extends BaseMessage<NotificationPayload> {
  type: 'notification';
}

export interface SystemAlertMessage extends BaseMessage<SystemAlertPayload> {
  type: 'system_alert';
}

export interface HeartbeatMessage extends BaseMessage<HeartbeatPayload> {
  type: 'heartbeat' | 'ping' | 'pong';
}

export interface AuthResponseMessage extends BaseMessage<AuthResponsePayload> {
  type: 'auth_response';
}

export interface DashboardUpdateMessage extends BaseMessage<DashboardUpdatePayload> {
  type: 'dashboard_update';
}

export interface ErrorMessage extends BaseMessage<ErrorPayload> {
  type: 'error';
}

export interface SubscribeMessage extends BaseMessage<SubscribePayload> {
  type: 'subscribe';
}

export interface UnsubscribeMessage extends BaseMessage<UnsubscribePayload> {
  type: 'unsubscribe';
}

// ============================================================================
// Union Type for All Messages
// ============================================================================

export type WebSocketMessage =
  | ScanUpdateMessage
  | NotificationMessage
  | SystemAlertMessage
  | HeartbeatMessage
  | AuthResponseMessage
  | DashboardUpdateMessage
  | ErrorMessage
  | SubscribeMessage
  | UnsubscribeMessage;

// ============================================================================
// Event Handler Types
// ============================================================================

export type MessageHandler<T = WebSocketMessage> = (message: T) => void;

export type EventType =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'reconnecting'
  | 'message'
  | 'scan_update'
  | 'notification'
  | 'system_alert'
  | 'dashboard_update';

export interface WebSocketEventHandlers {
  onConnected?: () => void;
  onDisconnected?: (event: CloseEvent) => void;
  onError?: (error: Error | Event) => void;
  onReconnecting?: (attempt: number, maxAttempts: number) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onScanUpdate?: (message: ScanUpdateMessage) => void;
  onNotification?: (message: NotificationMessage) => void;
  onSystemAlert?: (message: SystemAlertMessage) => void;
  onDashboardUpdate?: (message: DashboardUpdateMessage) => void;
}

// ============================================================================
// Connection Info
// ============================================================================

export interface ConnectionInfo {
  state: ConnectionState;
  connectedAt?: string;
  disconnectedAt?: string;
  reconnectAttempts: number;
  lastError?: string;
  latency?: number;
}

// ============================================================================
// Message Queue Item
// ============================================================================

export interface QueuedMessage {
  message: WebSocketMessage;
  timestamp: string;
  retries: number;
  priority: 'low' | 'normal' | 'high';
}

// ============================================================================
// Reconnection Info
// ============================================================================

export interface ReconnectionInfo {
  attempt: number;
  maxAttempts: number;
  nextAttemptIn: number;
  lastAttemptAt?: string;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isScanUpdateMessage(message: WebSocketMessage): message is ScanUpdateMessage {
  return message.type === 'scan_update';
}

export function isNotificationMessage(message: WebSocketMessage): message is NotificationMessage {
  return message.type === 'notification';
}

export function isSystemAlertMessage(message: WebSocketMessage): message is SystemAlertMessage {
  return message.type === 'system_alert';
}

export function isHeartbeatMessage(message: WebSocketMessage): message is HeartbeatMessage {
  return message.type === 'heartbeat' || message.type === 'ping' || message.type === 'pong';
}

export function isAuthResponseMessage(message: WebSocketMessage): message is AuthResponseMessage {
  return message.type === 'auth_response';
}

export function isDashboardUpdateMessage(message: WebSocketMessage): message is DashboardUpdateMessage {
  return message.type === 'dashboard_update';
}

export function isErrorMessage(message: WebSocketMessage): message is ErrorMessage {
  return message.type === 'error';
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a WebSocket message with proper structure
 */
export function createMessage<T>(
  type: MessageType,
  payload: T,
  id?: string
): BaseMessage<T> {
  return {
    type,
    payload,
    timestamp: new Date().toISOString(),
    id: id || crypto.randomUUID(),
  };
}

/**
 * Validate WebSocket message structure
 */
export function isValidMessage(data: unknown): data is WebSocketMessage {
  if (typeof data !== 'object' || data === null) {
    return false;
  }

  const message = data as Partial<WebSocketMessage>;
  
  return (
    typeof message.type === 'string' &&
    message.payload !== undefined &&
    typeof message.timestamp === 'string' &&
    typeof message.id === 'string'
  );
}

/**
 * Parse WebSocket message from string
 */
export function parseMessage(data: string): WebSocketMessage | null {
  try {
    const parsed = JSON.parse(data);
    return isValidMessage(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Serialize WebSocket message to string
 */
export function serializeMessage(message: WebSocketMessage): string {
  return JSON.stringify(message);
}
