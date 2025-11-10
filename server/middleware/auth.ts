import { Request, Response, NextFunction } from 'express';
import { jwtAuthService } from '../services/jwt-auth-service';
import { sessionManager } from '../services/session-manager';
import { logger } from '../config';

// Extend Request interface to include user
export interface AuthenticatedRequest extends Request {
  user?: {
    userId: number;
    username: string;
    email?: string;
    role: string;
  };
  validatedData?: any;
}

/**
 * JWT Authentication middleware
 * Verifies JWT token and adds user info to request
 */
export function requireAuth(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers['authorization'];
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ 
      success: false,
      message: 'Authentication required. Please provide a valid JWT token.' 
    });
  }
  
  const token = authHeader.replace('Bearer ', '');
  
  jwtAuthService.verifyToken(token)
    .then(payload => {
      if (payload) {
        // Validate session security
        const clientIP = req.ip || req.connection.remoteAddress || 'unknown';
        const userAgent = req.get('User-Agent') || 'unknown';
        
        const securityCheck = sessionManager.validateSessionSecurity(token, clientIP, userAgent);
        
        if (!securityCheck.valid) {
          if (securityCheck.suspicious) {
            // Force logout for suspicious activity
            sessionManager.destroySession(token);
            
            return res.status(401).json({
              success: false,
              message: 'Session security violation detected. Please log in again.',
              code: 'SECURITY_VIOLATION',
              reason: securityCheck.reason
            });
          } else {
            return res.status(401).json({
              success: false,
              message: securityCheck.reason || 'Session validation failed',
              code: 'SESSION_INVALID'
            });
          }
        }
        
        // Update session activity
        sessionManager.updateSessionActivity(token);
        
        req.user = {
          userId: payload.userId,
          username: payload.username,
          email: payload.email,
          role: payload.role
        };
        next();
      } else {
        return res.status(401).json({ 
          success: false,
          message: 'Invalid or expired token' 
        });
      }
    })
    .catch((error: any) => {
      logger.error('Auth middleware error', { error: error.message });
      return res.status(401).json({ 
        success: false,
        message: 'Authentication failed' 
      });
    });
}

/**
 * Optional authentication middleware
 * Adds user info to request if token is present and valid, but doesn't require it
 */
export function optionalAuth(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers['authorization'];
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return next(); // Continue without authentication
  }
  
  const token = authHeader.replace('Bearer ', '');
  
  jwtAuthService.verifyToken(token)
    .then(payload => {
      if (payload) {
        // Update session activity for valid tokens
        sessionManager.updateSessionActivity(token);
        
        req.user = {
          userId: payload.userId,
          username: payload.username,
          email: payload.email,
          role: payload.role
        };
      }
      next();
    })
    .catch((error: any) => {
      logger.debug('Optional auth failed', { error: error.message });
      next(); // Continue without authentication
    });
}

/**
 * Role-based authorization middleware
 * Requires specific role(s) to access the endpoint
 */
export function requireRole(...roles: string[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }
    
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: 'Insufficient permissions'
      });
    }
    
    next();
  };
}

/**
 * Admin-only middleware
 */
export const requireAdmin = requireRole('admin');

/**
 * User or admin middleware
 */
export const requireUserOrAdmin = requireRole('user', 'admin');