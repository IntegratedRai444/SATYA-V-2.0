import { WebSocket } from 'ws';
import { IncomingMessage } from 'http';

export interface WebSocketMessage {
  type: string;
  data?: any;
  error?: string;
  timestamp?: number;
  requestId?: string;
  retryCount?: number;
}

export interface AuthenticatedWebSocket extends WebSocket {
  userId: string;
  username?: string;
  sessionId: string;
  ipAddress: string;
  isAlive: boolean;
  lastActivity: number;
  subscribedChannels: Set<string>;
  messageCount: number;
  lastMessageTime: number;
  clientId?: string;
  connectedAt?: number;
  messageQueue?: Array<WebSocketMessage & { retryCount: number }>;
  maxQueueSize?: number;
  flushTimeout?: NodeJS.Timeout;
}

import { WebSocketAuthenticatedRequest } from '../../types/auth';

export interface WebSocketStats {
  totalConnections: number;
  connectedUsers: number;
  averageConnectionsPerUser: number;
}

export interface ProcessingJob {
  id: string;
  userId: string;
  status: string;
  progress: number;
  result?: any;
  error?: string;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  metrics?: Record<string, any>;
  metadata?: Record<string, any>;
}
