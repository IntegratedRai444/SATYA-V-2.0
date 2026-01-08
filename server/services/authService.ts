import { db } from '../db';
import { eq, and, sql } from 'drizzle-orm';
import { users, sessions, type User, type Session } from '../db/schema';
import { JwtAuthService } from './auth/jwtAuthService';
import { logger } from '../config/logger';

export class AuthService {
  /**
   * Find user by email
   */
  static async findUserByEmail(email: string): Promise<User | null> {
    try {
      const [user] = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);
      
      return user || null;
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
      const [user] = await db
        .select()
        .from(users)
        .where(eq(users.id, id))
        .limit(1);
      
      return user || null;
    } catch (error) {
      logger.error('Error finding user by ID', { error });
      throw error;
    }
  }

  /**
   * Create a new session
   */
  static async createSession(sessionData: Omit<Session, 'id' | 'createdAt' | 'updatedAt'>): Promise<Session> {
    try {
      const [session] = await db
        .insert(sessions)
        .values({
          ...sessionData,
          createdAt: new Date(),
          updatedAt: new Date(),
          lastUsedAt: new Date(),
          isRevoked: false,
          metadata: {}
        } as any) // Type assertion to handle the SQL expression type
        .returning();
      
      return session;
    } catch (error) {
      logger.error('Error creating session', { error });
      throw error;
    }
  }

  /**
   * Revoke a session
   */
  static async revokeSession(sessionId: string): Promise<void> {
    try {
      await db
        .update(sessions)
        .set({ 
          isRevoked: true,
          revokedAt: new Date(),
          updatedAt: new Date() 
        })
        .where(eq(sessions.id, sessionId));
    } catch (error) {
      logger.error('Error revoking session', { error });
      throw error;
    }
  }

  /**
   * Find active session by token
   */
  static async findSessionByToken(refreshToken: string): Promise<Session | null> {
    try {
      const [session] = await db
        .select()
        .from(sessions)
        .where(
          and(
            eq(sessions.refreshToken, refreshToken),
            eq(sessions.isRevoked, false),
            sql`${sessions.expiresAt} > NOW()`
          )
        )
        .limit(1);
      
      return session || null;
    } catch (error) {
      logger.error('Error finding session by token', { error });
      throw error;
    }
  }

  /**
   * Update user's last login timestamp
   */
  static async updateLastLogin(userId: number, ipAddress?: string): Promise<void> {
    try {
      await db
        .update(users)
        .set({ 
          lastLogin: new Date(),
          lastIp: ipAddress,
          loginCount: sql`${users.loginCount} + 1` as any,
          failedLoginAttempts: 0,
          updatedAt: new Date()
        })
        .where(eq(users.id, userId));
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
      await db
        .update(users)
        .set({ 
          lastFailedLogin: new Date(),
          failedLoginAttempts: sql`${users.failedLoginAttempts} + 1` as any,
          lastIp: ipAddress,
          updatedAt: new Date()
        })
        .where(eq(users.email, email));
    } catch (error) {
      logger.error('Error recording failed login', { error });
      throw error;
    }
  }
}

export default AuthService;
