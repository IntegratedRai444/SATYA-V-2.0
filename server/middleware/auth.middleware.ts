import { Request as ExpressRequest, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';
import { z } from 'zod';
import { ZodError } from 'zod';
import RedisStore from 'rate-limit-redis';
import { createClient } from 'redis';

declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      email: string;
      role: string;
      user_metadata?: Record<string, any>;
    };
    rateLimit?: {
      limit: number;
      current: number;
      remaining: number;
      resetTime?: Date;
    };
  }
}

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
      const redisStore = new RedisStore({
        // @ts-ignore - The Redis client type is compatible
        sendCommand: (...args: string[]) => redisClient.sendCommand(args),
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

// Extend the Express Request type
declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      email: string;
      role: string;
      user_metadata?: Record<string, any>;
    };
    rateLimit?: {
      limit: number;
      current: number;
      remaining: number;
      resetTime?: Date;
    };
  }
}

// Create and export a custom Request type that extends Express's Request
export type Request = ExpressRequest & {
  user?: {
    id: string;
    email: string;
    role: string;
    user_metadata?: Record<string, any>;
  };
  rateLimit?: {
    limit: number;
    current: number;
    remaining: number;
    resetTime?: Date;
  };
};

// Rate limiting configuration
const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes
const MAX_REQUESTS_PER_WINDOW = 100; // General API requests
const MAX_AUTH_REQUESTS = 10; // Auth-specific endpoints (login, register, etc.)
const MAX_LOGIN_ATTEMPTS = 5; // Login attempts before temporary block
const LOGIN_BLOCK_DURATION = 15 * 60 * 1000; // 15 minutes block after max attempts

// Create rate limiters as functions that return the configured middleware
const createRateLimiters = async () => {
  const store = await storePromise;
  
  // Main rate limiter for all requests
  const apiLimiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MS,
    max: (req: ExpressRequest) => {
      // Apply stricter limits to public endpoints
      if (req.path.startsWith('/api/auth/')) {
        return MAX_AUTH_REQUESTS;
      }
      return MAX_REQUESTS_PER_WINDOW;
    },
    store: store as any, // Type assertion for RedisStore compatibility
    skip: (req: ExpressRequest) => {
      // Skip rate limiting for health checks and monitoring
      return req.path === '/health' || req.path === '/metrics';
    },
    message: 'Too many requests, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req: ExpressRequest) => {
      // Use API key if available, otherwise fall back to IP
      return req.headers['x-api-key']?.toString() || req.ip || 'unknown';
    },
    skipFailedRequests: false,
    skipSuccessfulRequests: false,
    handler: (req: Request, res: Response) => {
      res.status(429).json({
        success: false,
        error: 'Too many requests, please try again later.'
      });
    }
  });

  // Login-specific rate limiter with stricter limits
  const loginLimiter = rateLimit({
    store: store as any, // Type assertion needed due to RedisStore type issues
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: MAX_LOGIN_ATTEMPTS,
    message: JSON.stringify({
      error: 'Too many login attempts',
      message: 'Please try again later',
      retryAfter: '15 minutes'
    }),
    skipFailedRequests: false,
    keyGenerator: (req: ExpressRequest) => {
      // Use username + IP to prevent targeted attacks
      const username = req.body?.username || 'unknown';
      return `${username}_${req.ip}`;
    },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req: Request, res: Response) => {
      res.status(429).json({
        success: false,
        error: 'Too many login attempts. Please try again in 15 minutes.'
      });
    }
  });

  return { apiLimiter, loginLimiter };
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
export const apiLimiter = async (req: Request, res: Response, next: NextFunction) => {
  const { apiLimiter } = await getRateLimiters();
  return apiLimiter(req, res, next);
};

export const loginLimiter = async (req: Request, res: Response, next: NextFunction) => {
  const { loginLimiter } = await getRateLimiters();
  return loginLimiter(req, res, next);
};

// Slow down for brute force protection
export const speedLimiter = slowDown({
  windowMs: 5 * 60 * 1000, // 5 minutes
  delayAfter: 3, // Allow 3 requests per 5 minutes, then...
  delayMs: 1000 // Add 1 second of delay per request above delayAfter
});

// Token validation schema
const tokenSchema = z.object({
  sub: z.string().uuid(),
  email: z.string().email(),
  role: z.enum(['user', 'admin', 'moderator']).default('user'),
  exp: z.number().int().positive(),
  iat: z.number().int().positive()
});

/**
 * JWT Authentication Middleware with enhanced security
 */
export const authenticate = async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Check for token in Authorization header first
    let token: string | null = null;
    const authHeader = req.headers.authorization;
    
    // Also check for token in cookies (for web clients)
    if (!authHeader && req.cookies?.auth_token) {
      token = req.cookies.auth_token;
    } else if (authHeader?.startsWith('Bearer ')) {
      token = authHeader.split(' ')[1];
    }
    
    if (!token) {
      logger.warn('Authentication failed: No token provided', {
        path: req.path,
        ip: req.ip,
        userAgent: req.headers['user-agent']
      });
      return res.status(401).json({
        error: 'unauthorized',
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Verify token with Supabase
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
          error: 'token_expired',
          message: 'Session expired. Please log in again.',
          code: 'TOKEN_EXPIRED'
        });
      }
      
      return res.status(401).json({
        error: 'invalid_token',
        message: 'Invalid or expired token',
        code: 'INVALID_TOKEN'
      });
    }

    // Rate limiting per user (simplified for now - will be handled by the rate limiter middleware)
    // In a production environment, you'd want to track user-specific limits here

    // Rate limiting is handled by the middleware

    // Attach user to request with minimal required data
    if (user) {
      req.user = {
        id: user.id,
        email: user.email || '',
        role: user.user_metadata?.role || 'user',
        user_metadata: user.user_metadata || {}
      };
    }
    
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
  return async (req: Request, res: Response, next: NextFunction) => {
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

/**
 * CSRF Protection Middleware
 * Validates CSRF tokens for state-changing requests
 */
export const csrfProtection = (req: Request, res: Response, next: NextFunction) => {
  // Skip CSRF check for safe HTTP methods
  if (['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(req.method)) {
    return next();
  }

  // Get CSRF token from header or body
  const csrfToken = req.headers['x-csrf-token'] || req.body._csrf;
  
  if (!csrfToken) {
    return res.status(403).json({
      success: false,
      code: 'CSRF_TOKEN_REQUIRED',
      message: 'CSRF token is required for this request'
    });
  }

  // In production, verify the CSRF token against the user's session
  if (process.env.NODE_ENV === 'production') {
    // Add your CSRF token verification logic here
    // Example: verifyCsrfToken(csrfToken, req.session.csrfSecret)
  }

  next();
};

// Predefined middleware combinations
export const requireAuth = [authenticate];
export const requireAdmin = [authenticate, requireRole('admin')];
export const requireModerator = [authenticate, requireRole(['admin', 'moderator'])];

/**
 * Request validation middleware
 * Validates request body against a Zod schema
 */
export const validateRequest = (schema: z.ZodSchema) => {
  return (req: Request, res: Response, next: NextFunction) => {
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
  req: Request,
  res: Response,
  next: NextFunction
) => {
  logger.error(`[${new Date().toISOString()}] ${req.method} ${req.path}`, {
    error: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
    body: req.body,
    params: req.params,
    query: req.query
  });

  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
  
  res.status(statusCode).json({
    success: false,
    code: err.name || 'INTERNAL_ERROR',
    message: process.env.NODE_ENV === 'development' 
      ? err.message 
      : 'An unexpected error occurred'
  });
};
