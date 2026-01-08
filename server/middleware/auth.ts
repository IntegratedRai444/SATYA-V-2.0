import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { unauthorizedResponse } from '../utils/apiResponse';
import { logger } from '../config/logger';

// Define the user type that will be attached to the request
export interface AuthUser {
  id: string;
  email: string;
  role: string;
  user_metadata?: Record<string, any>;
  [key: string]: any; // Allow additional properties
}

// Extend the Express Request type to include user information
declare global {
  namespace Express {
    interface Request {
      user?: AuthUser;
    }
  }
}

/**
 * Verify JWT token from Authorization header
 */
export const authenticateToken = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    // Get token from Authorization header
    const authHeader = req.headers.authorization;
    const token = authHeader?.split(' ')[1]; // Bearer <token>

    if (!token) {
      return unauthorizedResponse(res, 'No token provided');
    }

    // Verify token and cast to our AuthUser type
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as AuthUser;
    
    // Add user to request object with required properties
    if (!decoded.id || !decoded.email || !decoded.role) {
      throw new Error('Invalid token payload: missing required fields');
    }
    
    req.user = {
      ...decoded, // Spread all properties first
      // Explicitly set required fields to ensure correct types
      id: decoded.id,
      email: decoded.email,
      role: decoded.role,
      user_metadata: decoded.user_metadata
    };
    
    // Continue to the next middleware/route handler
    next();
  } catch (error) {
    logger.error('Authentication error:', { 
      error: error instanceof Error ? error.message : 'Unknown error',
      path: req.path,
      method: req.method,
    });
    
    if (error instanceof jwt.TokenExpiredError) {
      return unauthorizedResponse(res, 'Token has expired');
    }
    
    if (error instanceof jwt.JsonWebTokenError) {
      return unauthorizedResponse(res, 'Invalid token');
    }
    
    return unauthorizedResponse(res, 'Authentication failed');
  }
};

/**
 * Role-based access control middleware
 */
export const authorize = (roles: string | string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.user) {
        return unauthorizedResponse(res, 'User not authenticated');
      }

      const userRoles = Array.isArray(req.user.role) ? req.user.role : [req.user.role];
      const requiredRoles = Array.isArray(roles) ? roles : [roles];
      
      const hasPermission = requiredRoles.some(role => userRoles.includes(role));
      
      if (!hasPermission) {
        return unauthorizedResponse(res, 'Insufficient permissions');
      }
      
      next();
    } catch (error) {
      logger.error('Authorization error:', { 
        error: error instanceof Error ? error.message : 'Unknown error',
        path: req.path,
        method: req.method,
      });
      return unauthorizedResponse(res, 'Authorization failed');
    }
  };
};

/**
 * Rate limiting middleware for authentication endpoints
 */
export const authRateLimiter = (req: Request, res: Response, next: NextFunction) => {
  // Implement rate limiting logic here
  // Example: Limit to 5 requests per 15 minutes per IP
  next();
};