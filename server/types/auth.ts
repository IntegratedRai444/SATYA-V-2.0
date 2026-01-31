import { Request } from 'express';
import { User } from '@supabase/supabase-js';

// Custom user interface for authenticated requests
export interface AuthenticatedUser {
  id: string;
  email: string;
  role?: string;
  email_verified?: boolean;
  user_metadata?: Record<string, unknown>;
}

declare module 'express' {
  interface Request {
    user?: AuthenticatedUser;
    apiVersion?: string;
    id?: string;
    nonce?: string;
    rateLimit?: {
      limit: number;
      current: number;
      remaining: number;
      resetTime?: Date;
    };
  }
}

// Type alias for convenience
export type AuthenticatedRequest = Request & {
  user?: AuthenticatedUser;
  rateLimit?: {
    limit: number;
    current: number;
    remaining: number;
    resetTime?: Date;
  };
};

// WebSocket authenticated request interface
export interface WebSocketAuthenticatedRequest {
  userId: string;
  username?: string;
  sessionId: string;
  ipAddress: string;
  token?: string;
  user?: User;
}
