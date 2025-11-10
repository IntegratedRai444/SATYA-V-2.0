import { logger, logSecurity } from '../config';
import { jwtAuthService, JWTPayload } from './jwt-auth-service';

interface SessionInfo {
  userId: number;
  username: string;
  email?: string;
  role: string;
  loginTime: Date;
  lastActivity: Date;
  ipAddress: string;
  userAgent: string;
  isActive: boolean;
}

interface SessionStats {
  totalActiveSessions: number;
  userSessions: Map<number, SessionInfo[]>;
  sessionsByToken: Map<string, SessionInfo>;
}

class SessionManager {
  private activeSessions: Map<string, SessionInfo> = new Map();
  private userSessions: Map<number, Set<string>> = new Map();
  private sessionCleanupInterval: NodeJS.Timeout;

  constructor() {
    // Clean up expired sessions every 15 minutes
    this.sessionCleanupInterval = setInterval(() => {
      this.cleanupExpiredSessions();
    }, 15 * 60 * 1000);

    logger.info('Session manager initialized');
  }

  /**
   * Create a new session when user logs in
   */
  async createSession(token: string, payload: JWTPayload, ipAddress: string, userAgent: string): Promise<void> {
    try {
      // Check for suspicious activity (too many sessions)
      const existingSessionCount = this.getUserSessionCount(payload.userId);
      const maxSessionsPerUser = 10; // Configurable limit
      
      if (existingSessionCount >= maxSessionsPerUser) {
        logger.warn('User exceeded maximum session limit', {
          userId: payload.userId,
          username: payload.username,
          currentSessions: existingSessionCount,
          maxAllowed: maxSessionsPerUser
        });
        
        // Clean up oldest sessions
        await this.cleanupOldestUserSessions(payload.userId, maxSessionsPerUser - 1);
      }

      const sessionInfo: SessionInfo = {
        userId: payload.userId,
        username: payload.username,
        email: payload.email,
        role: payload.role,
        loginTime: new Date(),
        lastActivity: new Date(),
        ipAddress,
        userAgent,
        isActive: true
      };

      // Store session
      this.activeSessions.set(token, sessionInfo);

      // Track user sessions
      if (!this.userSessions.has(payload.userId)) {
        this.userSessions.set(payload.userId, new Set());
      }
      this.userSessions.get(payload.userId)!.add(token);

      logSecurity('Session created', {
        userId: payload.userId,
        username: payload.username,
        ipAddress,
        userAgent,
        sessionCount: this.getUserSessionCount(payload.userId)
      });

      logger.debug('Session created successfully', {
        userId: payload.userId,
        totalSessions: this.activeSessions.size
      });
    } catch (error) {
      logger.error('Failed to create session', {
        error: (error as Error).message,
        userId: payload.userId
      });
    }
  }

  /**
   * Update session activity
   */
  updateSessionActivity(token: string): void {
    const session = this.activeSessions.get(token);
    if (session) {
      session.lastActivity = new Date();
      this.activeSessions.set(token, session);
    }
  }

  /**
   * Refresh session with new token
   */
  async refreshSession(oldToken: string, newToken: string): Promise<boolean> {
    try {
      const session = this.activeSessions.get(oldToken);
      if (!session) {
        return false;
      }

      // Update session with new token
      session.lastActivity = new Date();
      
      // Remove old token mapping
      this.activeSessions.delete(oldToken);
      
      // Add new token mapping
      this.activeSessions.set(newToken, session);

      // Update user sessions mapping
      const userTokens = this.userSessions.get(session.userId);
      if (userTokens) {
        userTokens.delete(oldToken);
        userTokens.add(newToken);
      }

      logSecurity('Session refreshed', {
        userId: session.userId,
        username: session.username,
        ipAddress: session.ipAddress
      });

      return true;
    } catch (error) {
      logger.error('Failed to refresh session', {
        error: (error as Error).message
      });
      return false;
    }
  }

  /**
   * Destroy a specific session (logout)
   */
  async destroySession(token: string): Promise<boolean> {
    try {
      const session = this.activeSessions.get(token);
      if (!session) {
        return false;
      }

      // Remove from active sessions
      this.activeSessions.delete(token);

      // Remove from user sessions
      const userTokens = this.userSessions.get(session.userId);
      if (userTokens) {
        userTokens.delete(token);
        if (userTokens.size === 0) {
          this.userSessions.delete(session.userId);
        }
      }

      // Blacklist the token
      await jwtAuthService.logout(token);

      logSecurity('Session destroyed', {
        userId: session.userId,
        username: session.username,
        ipAddress: session.ipAddress,
        sessionDuration: Date.now() - session.loginTime.getTime()
      });

      logger.debug('Session destroyed successfully', {
        userId: session.userId,
        totalSessions: this.activeSessions.size
      });

      return true;
    } catch (error) {
      logger.error('Failed to destroy session', {
        error: (error as Error).message,
        token: token.substring(0, 10) + '...'
      });
      return false;
    }
  }

  /**
   * Destroy all sessions for a user
   */
  async destroyAllUserSessions(userId: number, excludeToken?: string): Promise<number> {
    try {
      const userTokens = this.userSessions.get(userId);
      if (!userTokens) {
        return 0;
      }

      const tokensToDestroy = Array.from(userTokens).filter(token => token !== excludeToken);
      let destroyedCount = 0;

      for (const token of Array.from(tokensToDestroy)) {
        const success = await this.destroySession(token);
        if (success) {
          destroyedCount++;
        }
      }

      logSecurity('All user sessions destroyed', {
        userId,
        destroyedCount,
        excludedCurrentSession: !!excludeToken
      });

      return destroyedCount;
    } catch (error) {
      logger.error('Failed to destroy all user sessions', {
        error: (error as Error).message,
        userId
      });
      return 0;
    }
  }

  /**
   * Get session information
   */
  getSession(token: string): SessionInfo | null {
    return this.activeSessions.get(token) || null;
  }

  /**
   * Get all sessions for a user
   */
  getUserSessions(userId: number): SessionInfo[] {
    const userTokens = this.userSessions.get(userId);
    if (!userTokens) {
      return [];
    }

    const sessions: SessionInfo[] = [];
    for (const token of Array.from(userTokens)) {
      const session = this.activeSessions.get(token);
      if (session) {
        sessions.push(session);
      }
    }

    return sessions;
  }

  /**
   * Get session count for a user
   */
  getUserSessionCount(userId: number): number {
    const userTokens = this.userSessions.get(userId);
    return userTokens ? userTokens.size : 0;
  }

  /**
   * Check if user has active sessions
   */
  hasActiveSessions(userId: number): boolean {
    return this.getUserSessionCount(userId) > 0;
  }

  /**
   * Get session statistics
   */
  getSessionStats(): SessionStats {
    const userSessionsMap = new Map<number, SessionInfo[]>();
    const sessionsByTokenMap = new Map<string, SessionInfo>();

    for (const [token, session] of Array.from(this.activeSessions.entries())) {
      // Group by user
      if (!userSessionsMap.has(session.userId)) {
        userSessionsMap.set(session.userId, []);
      }
      userSessionsMap.get(session.userId)!.push(session);

      // Map by token
      sessionsByTokenMap.set(token, session);
    }

    return {
      totalActiveSessions: this.activeSessions.size,
      userSessions: userSessionsMap,
      sessionsByToken: sessionsByTokenMap
    };
  }

  /**
   * Clean up expired sessions
   */
  private cleanupExpiredSessions(): void {
    const now = Date.now();
    const maxInactiveTime = 24 * 60 * 60 * 1000; // 24 hours
    let cleanedCount = 0;

    for (const [token, session] of Array.from(this.activeSessions.entries())) {
      const inactiveTime = now - session.lastActivity.getTime();
      
      if (inactiveTime > maxInactiveTime) {
        this.destroySession(token);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      logger.info('Cleaned up expired sessions', {
        cleanedCount,
        remainingSessions: this.activeSessions.size
      });
    }
  }

  /**
   * Force cleanup of all sessions (for shutdown)
   */
  async cleanup(): Promise<void> {
    clearInterval(this.sessionCleanupInterval);
    
    const sessionCount = this.activeSessions.size;
    this.activeSessions.clear();
    this.userSessions.clear();

    logger.info('Session manager cleanup completed', {
      clearedSessions: sessionCount
    });
  }

  /**
   * Clean up oldest sessions for a user
   */
  private async cleanupOldestUserSessions(userId: number, keepCount: number): Promise<void> {
    const userSessions = this.getUserSessions(userId);
    if (userSessions.length <= keepCount) {
      return;
    }

    // Sort by login time (oldest first)
    const sortedSessions = userSessions.sort((a, b) => a.loginTime.getTime() - b.loginTime.getTime());
    const sessionsToRemove = sortedSessions.slice(0, sortedSessions.length - keepCount);

    for (const session of sessionsToRemove) {
      // Find token for this session
      for (const [token, sessionInfo] of this.activeSessions.entries()) {
        if (sessionInfo === session) {
          await this.destroySession(token);
          break;
        }
      }
    }

    logger.info('Cleaned up oldest user sessions', {
      userId,
      removedSessions: sessionsToRemove.length,
      remainingSessions: keepCount
    });
  }

  /**
   * Validate session security
   */
  validateSessionSecurity(token: string, currentIp: string, currentUserAgent: string): {
    valid: boolean;
    reason?: string;
    suspicious?: boolean;
  } {
    const session = this.activeSessions.get(token);
    if (!session) {
      return { valid: false, reason: 'Session not found' };
    }

    // Check if session is still active
    if (!session.isActive) {
      return { valid: false, reason: 'Session is inactive' };
    }

    // Check for IP address changes (potential session hijacking)
    if (session.ipAddress !== currentIp) {
      logSecurity('Suspicious session activity - IP change detected', {
        userId: session.userId,
        username: session.username,
        originalIp: session.ipAddress,
        currentIp,
        sessionAge: Date.now() - session.loginTime.getTime()
      });

      return { 
        valid: false, 
        reason: 'IP address mismatch', 
        suspicious: true 
      };
    }

    // Check for significant user agent changes
    if (session.userAgent !== currentUserAgent) {
      // Allow minor differences but flag major changes
      const similarity = this.calculateUserAgentSimilarity(session.userAgent, currentUserAgent);
      if (similarity < 0.7) {
        logSecurity('Suspicious session activity - User Agent change detected', {
          userId: session.userId,
          username: session.username,
          originalUserAgent: session.userAgent,
          currentUserAgent,
          similarity
        });

        return { 
          valid: false, 
          reason: 'User agent mismatch', 
          suspicious: true 
        };
      }
    }

    return { valid: true };
  }

  /**
   * Calculate similarity between user agent strings
   */
  private calculateUserAgentSimilarity(ua1: string, ua2: string): number {
    if (ua1 === ua2) return 1.0;
    
    const words1 = ua1.toLowerCase().split(/\s+/);
    const words2 = ua2.toLowerCase().split(/\s+/);
    
    const commonWords = words1.filter(word => words2.includes(word));
    const totalWords = new Set([...words1, ...words2]).size;
    
    return commonWords.length / totalWords;
  }

  /**
   * Get session activity report
   */
  getActivityReport(): {
    totalSessions: number;
    activeUsers: number;
    averageSessionsPerUser: number;
    oldestSession: Date | null;
    newestSession: Date | null;
  } {
    const sessions = Array.from(this.activeSessions.values());
    const uniqueUsers = new Set(sessions.map(s => s.userId)).size;
    
    let oldestSession: Date | null = null;
    let newestSession: Date | null = null;

    if (sessions.length > 0) {
      const loginTimes = sessions.map(s => s.loginTime);
      oldestSession = new Date(Math.min(...loginTimes.map(d => d.getTime())));
      newestSession = new Date(Math.max(...loginTimes.map(d => d.getTime())));
    }

    return {
      totalSessions: sessions.length,
      activeUsers: uniqueUsers,
      averageSessionsPerUser: uniqueUsers > 0 ? sessions.length / uniqueUsers : 0,
      oldestSession,
      newestSession
    };
  }
}

// Export singleton instance
export const sessionManager = new SessionManager();

// Graceful shutdown handler
process.on('SIGTERM', async () => {
  await sessionManager.cleanup();
});

process.on('SIGINT', async () => {
  await sessionManager.cleanup();
});