import { dbManager } from '../db';
import { eq, and, sql } from 'drizzle-orm';
import { users } from '@shared/schema';

// Define User type matching the database schema
type User = {
  id: number;
  username: string;
  password: string;
  email: string | null;
  full_name: string | null;
  api_key: string | null;
  role: string;
  failed_login_attempts: number;
  last_failed_login: string | null;
  is_locked: boolean;
  lockout_until: string | null;
  created_at: string;
  updated_at: string;
  last_login?: string | null;
};

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
import { JwtAuthService } from './auth/jwtAuthService';
import { logger } from '../config/logger';

export class AuthService {
  /**
   * Find user by email
   */
  static async findUserByEmail(email: string): Promise<User | null> {
    try {
      const users = await dbManager.find('users', { email }, { limit: 1 }) as User[];
      return users[0] || null;
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
      const user = await dbManager.findById('users', id.toString()) as User | null;
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
    // In a real implementation, you would store the session in a database
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
    // In a real implementation, you would update the session in the database
    logger.info(`Session ${sessionId} would be revoked in a real implementation`);
  }

  /**
   * Find active session by token (stub implementation)
   */
  static async findSessionByToken(refreshToken: string): Promise<Session | null> {
    // In a real implementation, you would look up the session in the database
    // For now, we'll return null to indicate no session was found
    logger.debug(`Looking up session for token: ${refreshToken.substring(0, 10)}...`);
    return null;
  }

  /**
   * Update user's last login timestamp
   */
  static async updateLastLogin(userId: number, ipAddress?: string): Promise<void> {
    try {
      // Only update the fields that exist in the database schema
      await dbManager.update('users', userId.toString(), { 
        last_failed_login: null,
        failed_login_attempts: 0,
        updated_at: new Date().toISOString()
      });
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
      const user = (await dbManager.find('users', { email }, { limit: 1 }))[0];
      if (!user) return;
      
      const failedAttempts = (user.failed_login_attempts || 0) + 1;
      
      await dbManager.update('users', user.id.toString(), {
        last_failed_login: new Date().toISOString(),
        failed_login_attempts: failedAttempts,
        updated_at: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Error recording failed login', { error });
      throw error;
    }
  }
}

export default AuthService;
