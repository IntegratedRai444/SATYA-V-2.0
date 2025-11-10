import { Request, Response, NextFunction } from 'express';
import { sessionManager } from '../services/session-manager';
import { jwtAuthService } from '../services/jwt-auth-service';
import { logger } from '../config';

/**
 * Middleware to track session activity for authenticated requests
 */
export function trackSessionActivity(req: Request, res: Response, next: NextFunction) {
  // Extract token from request
  const authHeader = req.headers.authorization;
  
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.replace('Bearer ', '');
    
    // Update session activity asynchronously (don't block request)
    setImmediate(async () => {
      try {
        // Verify token is still valid
        const payload = await jwtAuthService.verifyToken(token);
        if (payload) {
          sessionManager.updateSessionActivity(token);
        }
      } catch (error) {
        // Silently handle errors - don't affect the main request
        logger.debug('Session activity tracking failed', {
          error: (error as Error).message,
          token: token.substring(0, 10) + '...'
        });
      }
    });
  }
  
  next();
}

/**
 * Middleware to check for session limits per user
 */
export function checkSessionLimits(maxSessionsPerUser: number = 5) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const authHeader = req.headers.authorization;
    
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const token = authHeader.replace('Bearer ', '');
      
      try {
        const payload = await jwtAuthService.verifyToken(token);
        if (payload) {
          const sessionCount = sessionManager.getUserSessionCount(payload.userId);
          
          if (sessionCount > maxSessionsPerUser) {
            logger.warn('User exceeded session limit', {
              userId: payload.userId,
              username: payload.username,
              sessionCount,
              maxAllowed: maxSessionsPerUser
            });
            
            return res.status(429).json({
              success: false,
              message: `Too many active sessions. Maximum ${maxSessionsPerUser} sessions allowed.`,
              code: 'SESSION_LIMIT_EXCEEDED'
            });
          }
        }
      } catch (error) {
        // If token verification fails, let other middleware handle it
        logger.debug('Session limit check failed', {
          error: (error as Error).message
        });
      }
    }
    
    next();
  };
}

/**
 * Middleware to add session info to response headers
 */
export function addSessionHeaders(req: Request, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.replace('Bearer ', '');
    const session = sessionManager.getSession(token);
    
    if (session) {
      res.setHeader('X-Session-Active', 'true');
      res.setHeader('X-Session-Login-Time', session.loginTime.toISOString());
      res.setHeader('X-Session-Last-Activity', session.lastActivity.toISOString());
    }
  }
  
  next();
}