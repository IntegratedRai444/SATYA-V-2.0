import { Request as ExpressRequest, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';
import { z } from 'zod';
import { ZodError } from 'zod';
import RedisStore from 'rate-limit-redis';
import { createClient } from 'redis';
import { AuthenticatedRequest, AuthenticatedUser } from '../types/auth';
import { User } from '@supabase/supabase-js';

// In-memory store for rate limiting when Redis is not available
class MemoryStore {
  private hits: Map<string, { count: number; resetTime: number }> = new Map();
  private windowMs: number;

  constructor(options: { windowMs: number }) {
    this.windowMs = options.windowMs;
  }

  async increment(key: string): Promise<{ totalHits: number; resetTime: Date }> {
    const now = Date.now();
    const resetTime = new Date(now + this.windowMs);
    
    const hit = this.hits.get(key);
    if (!hit || now > hit.resetTime) {
      this.hits.set(key, { count: 1, resetTime: now + this.windowMs });
      return { totalHits: 1, resetTime };
    }
    
    hit.count++;
    return { totalHits: hit.count, resetTime };
  }

  async decrement(key: string): Promise<void> {
    const hit = this.hits.get(key);
    if (hit) {
      hit.count = Math.max(0, hit.count - 1);
    }
  }

  async resetKey(key: string): Promise<void> {
    this.hits.delete(key);
  }
}

// Initialize rate limiting store
let rateLimitStore: MemoryStore | RedisStore;

// Create a function to initialize the rate limit store
const initializeRateLimitStore = async () => {
  if (process.env.REDIS_URL) {
    try {
      const redisClient = createClient({
        url: process.env.REDIS_URL,
        socket: {
          reconnectStrategy: (retries) => Math.min(retries * 100, 5000)
        }
      });

      redisClient.on('error', (err) => {
        logger.error('Redis error:', err);
        // Fall back to in-memory store
        rateLimitStore = new MemoryStore({ windowMs: 15 * 60 * 1000 });
      });

      await redisClient.connect();
      
      // Create Redis store with proper typing
      // Note: Type incompatibility between ioredis and rate-limit-redis
      const redisStore = new RedisStore({
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
sendCommand: (...args: unknown[]) => (redisClient as any).sendCommand(...args),
        prefix: 'ratelimit:'
      });
      
      rateLimitStore = redisStore;
      return redisStore;
    } catch (error) {
      logger.error('Failed to initialize Redis store, falling back to in-memory store:', error);
      rateLimitStore = new MemoryStore({ windowMs: 15 * 60 * 1000 });
      return rateLimitStore;
    }
  } else {
    rateLimitStore = new MemoryStore({ windowMs: 15 * 60 * 1000 });
    return rateLimitStore;
  }
};

// Initialize the store immediately and store the promise
const storePromise = initializeRateLimitStore();

// Auth state cache to prevent repeated Supabase calls
const authCache = new Map<string, { user: User; expires: number }>();

// Rate limiting configuration
const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes
const MAX_REQUESTS_PER_WINDOW = 100; // General API requests
const MAX_AUTH_REQUESTS = 10; // Auth-specific endpoints (login, register, etc.)
const MAX_REFRESH_REQUESTS = 5; // Refresh token requests per window

// Create rate limiters as functions that return the configured middleware
const createRateLimiters = async () => {
  const store = await storePromise;
  
  // Main rate limiter for all requests
  const apiRateLimiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MS,
    max: MAX_REQUESTS_PER_WINDOW,
    store: store as RedisStore | MemoryStore, // Type assertion for RedisStore compatibility
    skip: (req: ExpressRequest) => {
      // Skip rate limiting for health checks and static files
      return req.path === '/health' || req.path.startsWith('/static/');
    },
    keyGenerator: (req) => {
      return (req.ip || 'unknown') + ':' + (req.user?.id || 'anonymous');
    },
    handler: (req: ExpressRequest, res: Response) => {
      res.status(429).json({
        success: false,
        code: 'RATE_LIMIT_EXCEEDED',
        message: 'Too many requests, please try again later.',
        retryAfter: Math.ceil(RATE_LIMIT_WINDOW_MS / 1000)
      });
    }
  });

  const authRateLimiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MS,
    max: MAX_AUTH_REQUESTS,
    message: 'Too many authentication attempts, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
    store: store as RedisStore | MemoryStore, // Type assertion needed due to RedisStore type issues
    keyGenerator: (req: ExpressRequest) => {
      // Include both IP and email if available for auth endpoints
      const identifier = req.body?.email || 'unknown';
      return `auth:${req.ip || 'unknown'}:${identifier}`;
    },
    skip: (req) => {
      // Skip rate limiting for password reset requests
      return req.path.includes('reset-password');
    },
    handler: (req: ExpressRequest, res: Response) => {
      res.status(429).json({
        success: false,
        code: 'AUTH_RATE_LIMIT_EXCEEDED',
        message: 'Too many authentication attempts, please try again later.',
        retryAfter: Math.ceil(RATE_LIMIT_WINDOW_MS / 1000)
      });
    }
  });

  // Dedicated rate limiter for refresh tokens
  const refreshTokenRateLimiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MS,
    max: MAX_REFRESH_REQUESTS,
    message: 'Too many refresh token attempts, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
    store: store as RedisStore | MemoryStore, // Type assertion needed due to RedisStore type issues
    keyGenerator: (req: ExpressRequest) => {
      // Use IP + user ID if available, otherwise just IP
      const userId = req.user?.id || 'anonymous';
      const refreshToken = req.body?.refresh_token || 'none';
      return `refresh:${req.ip || 'unknown'}:${userId}:${refreshToken.substring(0, 10)}`;
    },
    handler: (req: ExpressRequest, res: Response) => {
      const retryAfter = Math.ceil(RATE_LIMIT_WINDOW_MS / 1000);
      res.status(429).json({
        success: false,
        code: 'REFRESH_RATE_LIMIT_EXCEEDED',
        message: `Too many refresh attempts. Please try again in ${retryAfter} seconds.`,
        retryAfter
      });
    }
  });

  return {
    apiRateLimiter,
    authRateLimiter,
    refreshTokenRateLimiter
  };
};

// Initialize rate limiters
let rateLimiters: Awaited<ReturnType<typeof createRateLimiters>>;

// Export rate limiters with initialization
export const getRateLimiters = async () => {
  if (!rateLimiters) {
    rateLimiters = await createRateLimiters();
  }
  return rateLimiters;
};

// Export rate limiters with initialization wrapper
export const apiLimiter = async (req: ExpressRequest, res: Response, next: NextFunction) => {
  const { apiRateLimiter } = await getRateLimiters();
  return apiRateLimiter(req as unknown as Parameters<typeof apiRateLimiter>[0], res, next);
};

export const loginLimiter = async (req: ExpressRequest, res: Response, next: NextFunction) => {
  const { authRateLimiter } = await getRateLimiters();
  return authRateLimiter(req as unknown as Parameters<typeof authRateLimiter>[0], res, next);
};

export const refreshTokenLimiter = async (req: ExpressRequest, res: Response, next: NextFunction) => {
  const { refreshTokenRateLimiter } = await getRateLimiters();
  return refreshTokenRateLimiter(req as unknown as Parameters<typeof refreshTokenRateLimiter>[0], res, next);
};

// Slow down for brute force protection
export const speedLimiter = slowDown({
  windowMs: 5 * 60 * 1000, // 5 minutes
  delayAfter: 3, // Allow 3 requests per 5 minutes, then...
  delayMs: () => 1000, // Add 1 second of delay per request above delayAfter
  validate: { delayMs: false } // Disable validation warning
});

// Helper function to safely validate user authentication
export const validateAuth = (req: AuthenticatedRequest, res: Response): boolean => {
  if (!req.user || !req.user.id) {
    res.status(401).json({ 
      success: false, 
      error: "Unauthorized" 
    });
    return false;
  }
  return true;
};

// Helper function to get safe user ID
export const getUserId = (req: AuthenticatedRequest): string | null => {
  return req.user?.id || null;
};

/**
 * Supabase Authentication Middleware with enhanced security and email verification
 */
export const authenticate = async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  try {
    // Debug: Log the authentication attempt
    logger.debug('Authentication attempt', {
      path: req.path,
      hasAuthHeader: !!req.headers.authorization,
      hasCookie: !!req.cookies?.['sb-access-token'],
      userAgent: req.headers['user-agent']
    });

    // Check for token in Authorization header first
    let token: string | null = null;
    const authHeader = req.headers.authorization;
    
    // Also check for token in cookies (for web clients)
    if (!authHeader && req.cookies?.['sb-access-token']) {
      token = req.cookies['sb-access-token'];
    } else if (authHeader?.startsWith('Bearer ')) {
      token = authHeader.substring(7); // Remove 'Bearer ' prefix
    }
    
    if (!token) {
      logger.warn('Authentication failed: No token provided', {
        path: req.path,
        ip: req.ip,
        userAgent: req.headers['user-agent']
      });
      return res.status(401).json({
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Check cache first to prevent repeated Supabase calls
    if (authCache.has(token)) {
      const cached = authCache.get(token)!;
      if (Date.now() < cached.expires) {
        // Convert Supabase User to AuthenticatedUser
        const authenticatedUser: AuthenticatedUser = {
          id: cached.user.id,
          email: cached.user.email || '',
          role: cached.user.user_metadata?.role as string,
          email_verified: cached.user.email_confirmed_at != null,
          user_metadata: cached.user.user_metadata || {}
        };
        req.user = authenticatedUser;
        return next();
      } else {
        // Cache expired, remove it
        authCache.delete(token);
      }
    }

    // Verify token with Supabase first (for user info)
    logger.debug('Verifying token with Supabase', { tokenLength: token?.length });
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error) {
      logger.warn('Token verification failed', {
        error: error.message,
        path: req.path,
        userId: user?.id
      });
      
      // Check for token expiration specifically
      if (error.message.includes('expired')) {
        return res.status(401).json({
          success: false,
          error: 'TOKEN_EXPIRED',
          message: 'Session expired. Please log in again.',
          code: 'TOKEN_EXPIRED'
        });
      }
      
      return res.status(401).json({
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'Invalid authentication token',
        code: 'AUTH_REQUIRED'
      });
    }

    if (!user) {
      logger.warn('Token verification failed: No user returned', {
        path: req.path
      });
      return res.status(401).json({
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'Invalid authentication token',
        code: 'AUTH_REQUIRED'
      });
    }

    // Attach user to request with minimal required data
    const authenticatedUser: AuthenticatedUser = {
      id: user.id,
      email: user.email || '',
      role: user.user_metadata?.role || 'user',
      email_verified: !!user.email_confirmed_at,
      user_metadata: user.user_metadata || {}
    };
    req.user = authenticatedUser;
    
    // Set security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    
    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    return res.status(500).json({ 
      success: false,
      code: 'AUTH_ERROR',
      message: 'Internal server error during authentication' 
    });
  }
};

/**
 * Role-Based Access Control Middleware
 * @param roles Array of allowed roles or a single role
 */
// Role type for type safety
type UserRole = 'user' | 'admin' | 'moderator';

export const requireRole = (roles: UserRole | UserRole[]) => {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ 
        success: false,
        code: 'UNAUTHORIZED',
        message: 'Authentication required' 
      });
    }

    try {
      // Default to 'user' role if not specified
      const userRole = req.user.role || 'user';
      const userRoles = Array.isArray(userRole) ? userRole : [userRole];
      const requiredRoles = Array.isArray(roles) ? roles : [roles];
      
      const hasRole = requiredRoles.some(role => userRoles.includes(role));
      
      if (!hasRole) {
        return res.status(403).json({ 
          success: false,
          code: 'FORBIDDEN',
          message: 'Insufficient permissions' 
        });
      }
      
      next();
    } catch (error) {
      logger.error('Role check error:', error);
      return res.status(500).json({ 
        success: false,
        code: 'ROLE_CHECK_ERROR',
        message: 'Error checking user permissions' 
      });
    }
  };
};

// CSRF protection is not needed for JWT-based authentication
// since we use Authorization headers instead of cookies
export const generateCsrfToken = () => {
  return { token: '', cookie: '' };
};

export const csrfProtection = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  return next();
};

/**
 * Middleware to check if user's email is verified
 * Must be used after authenticate middleware
 */
export const requireVerifiedEmail = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  if (!req.user) {
    return res.status(401).json({
      success: false,
      code: 'AUTH_REQUIRED',
      message: 'Authentication required'
    });
  }

  if (!req.user.email_verified) {
    return res.status(403).json({
      success: false,
      code: 'EMAIL_NOT_VERIFIED',
      message: 'Please verify your email address to access this resource'
    });
  }

  next();
};

// Predefined middleware combinations
export const requireAuth = [authenticate];
export const requireVerifiedUser = [authenticate, requireVerifiedEmail];
export const requireAdmin = [authenticate, requireRole('admin')];
export const requireModerator = [authenticate, requireRole(['admin', 'moderator'])];

/**
 * Request validation middleware
 * Validates request body against a Zod schema
 */
export const validateRequest = (schema: z.ZodSchema) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          success: false,
          code: 'VALIDATION_ERROR',
          message: 'Invalid request data',
          errors: error.issues
        });
      }
      next(error);
    }
  };
};

/**
 * Error handling middleware
 * Standardizes error responses
 */
export const errorHandler = (
  err: Error,
  req: AuthenticatedRequest,
  res: Response,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _next: NextFunction
) => {
  logger.error(`[${new Date().toISOString()}] ${req.method} ${req.path}`, {
    error: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
    body: req.body,
    params: req.params,
    query: req.query
  });

  // Determine appropriate status code based on error type
  let responseStatusCode = 500;
  let errorMessage = 'Internal Server Error';

  if (err.name === 'ValidationError') {
    responseStatusCode = 400;
    errorMessage = 'Bad Request';
  } else if (err.name === 'UnauthorizedError') {
    responseStatusCode = 401;
    errorMessage = 'Unauthorized';
  } else if (err.name === 'ForbiddenError') {
    responseStatusCode = 403;
    errorMessage = 'Forbidden';
  } else if (err.name === 'NotFoundError') {
    responseStatusCode = 404;
    errorMessage = 'Not Found';
  } else if (err.name === 'ServiceUnavailableError') {
    responseStatusCode = 503;
    errorMessage = 'Service Temporarily Unavailable';
  }

  const finalStatusCode = res.statusCode === 200 ? responseStatusCode : res.statusCode;
  
  res.status(finalStatusCode).json({
    success: false,
    code: err.name || 'INTERNAL_ERROR',
    message: process.env.NODE_ENV === 'development' 
      ? err.message 
      : errorMessage
  });
};
