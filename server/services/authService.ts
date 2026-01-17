import { supabase } from '../config/supabase';
import type { User } from '@shared/schema';
import type { Database } from '@shared/supabase.types';
import { JwtAuthService } from './auth/jwtAuthService';
import { logger } from '../config/logger';

export class AuthService {
  /**
   * Find user by email
   */
  static async findUserByEmail(email: string): Promise<User | null> {
    try {
      const { data: users, error } = await supabase
        .from('users')
        .select('*')
        .eq('email', email)
        .limit(1);
      
      if (error) {
        logger.error('Error finding user by email', { error });
        throw error;
      }
      
      return users && users.length > 0 ? users[0] : null;
    } catch (error) {
      logger.error('Error finding user by email', { error });
      throw error;
    }
  }

  /**
   * Find user by ID
   */
  static async findUserById(id: number): Promise<User | null> {
    try {
      const { data: user, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', id)
        .single();
      
      if (error) {
        logger.error('Error finding user by ID', { error });
        throw error;
      }
      
      return user;
    } catch (error) {
      logger.error('Error finding user by ID', { error });
      throw error;
    }
  }

  /**
   * Create a new session (stub implementation)
   * Note: Session management is handled by JWT tokens in this implementation
   */
  static async createSession(sessionData: Omit<Session, 'id' | 'createdAt' | 'updatedAt'>): Promise<Session> {
    // In a real implementation, you would store session in a database
    // For now, we'll return a session object with the provided data
    const now = new Date();
    return {
      ...sessionData,
      id: 'session-' + Math.random().toString(36).substr(2, 9),
      createdAt: now,
      updatedAt: now,
      lastUsedAt: now,
      isRevoked: false,
      metadata: {}
    };
  }

  /**
   * Revoke a session (stub implementation)
   */
  static async revokeSession(sessionId: string): Promise<void> {
    // In a real implementation, you would update session in database
    logger.info(`Session ${sessionId} would be revoked in a real implementation`);
  }

  /**
   * Find active session by token (stub implementation)
   */
  static async findSessionByToken(refreshToken: string): Promise<Session | null> {
    // In a real implementation, you would look up session in database
    // For now, we'll return null to indicate no session was found
    logger.debug(`Looking up session for token: ${refreshToken.substring(0, 10)}...`);
    return null;
  }

  /**
   * Update user's last login timestamp
   */
  static async updateLastLogin(userId: number, ipAddress?: string): Promise<void> {
    try {
      // Only update fields that exist in database schema
      const { error } = await supabase
        .from('users')
        .update({ 
          last_failed_login: null,
          failed_login_attempts: 0,
          updated_at: new Date().toISOString()
        })
        .eq('id', userId);
      
      if (error) {
        logger.error('Error updating last login', { error });
        throw error;
      }
    } catch (error) {
      logger.error('Error updating last login', { error });
      throw error;
    }
  }

  /**
   * Record failed login attempt
   */
  static async recordFailedLogin(email: string, ipAddress?: string): Promise<void> {
    try {
      // First get the current user to get the current failed attempts count
      const { data: users, error: fetchError } = await supabase
        .from('users')
        .select('*')
        .eq('email', email)
        .limit(1);
      
      if (fetchError) throw fetchError;
      if (!users || users.length === 0) return;
      
      const user = users[0] as User;
      const failedAttempts = (user.failed_login_attempts || 0) + 1;
      
      const { error: updateError } = await supabase
        .from('users')
        .update({
          last_failed_login: new Date().toISOString(),
          failed_login_attempts: failedAttempts,
          updated_at: new Date().toISOString()
        })
        .eq('id', user.id);
      
      if (updateError) {
        logger.error('Error recording failed login', { error: updateError });
        throw updateError;
      }
    } catch (error) {
      logger.error('Error recording failed login', { error });
      throw error;
    }
  }
}

// Define Session type
type Session = {
  id: string;
  userId: number;
  refreshToken: string;
  userAgent?: string;
  ipAddress?: string;
  expiresAt: Date;
  isRevoked: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastUsedAt: Date;
  metadata: Record<string, any>;
  revokedAt?: Date;
};
