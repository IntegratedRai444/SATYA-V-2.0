import { Request, Response, NextFunction } from 'express';
import { verifyAuthHeader } from '../config/supabase';
import { logger } from '../config/logger';

/**
 * Middleware to verify Supabase JWT token from Authorization header
 */
export const supabaseAuth = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const authHeader = req.headers.authorization || req.headers.Authorization;
    
    if (!authHeader) {
      return res.status(401).json({ 
        success: false, 
        error: 'Authorization header is required' 
      });
    }

    const { valid, user, error } = await verifyAuthHeader(
      Array.isArray(authHeader) ? authHeader[0] : authHeader
    );

    if (!valid || !user) {
      return res.status(401).json({ 
        success: false, 
        error: error || 'Invalid or expired token' 
      });
    }

    // Ensure required fields are present
    if (!user.email) {
      return res.status(401).json({
        success: false,
        error: 'Invalid user data: email is required'
      });
    }

    // Create a properly typed user object that matches Express.Request['user'] type
    const { id, email, role, user_metadata, ...rest } = user;
    const authenticatedUser = {
      id: id || '',
      email: email, // We've already checked this exists
      role: role || 'user',
      email_verified: user_metadata?.email_verified || false, // Default to false if not provided
      user_metadata: user_metadata || {},
      // Include other Supabase user properties
      ...rest
    };

    // Attach user to request object
    req.user = authenticatedUser;
    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    return res.status(500).json({ 
      success: false, 
      error: 'Internal server error during authentication' 
    });
  }
};

/**
 * Middleware to require authentication for specific routes
 */
export const requireAuth = (req: Request, res: Response, next: NextFunction) => {
  if (!req.user) {
    return res.status(401).json({ 
      success: false, 
      error: 'Authentication required' 
    });
  }
  next();
};

/**
 * Middleware to check for admin role
 */
export const requireAdmin = (req: Request, res: Response, next: NextFunction) => {
  if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({ 
      success: false, 
      error: 'Admin privileges required' 
    });
  }
  next();
};

export default {
  supabaseAuth,
  requireAuth,
  requireAdmin
};
