import { Request as ExpressRequest, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';
import { z } from 'zod';
import { ZodError } from 'zod';
import RedisStore from 'rate-limit-redis';
import { createClient } from 'redis';
import crypto from 'crypto';

// Node.js globals
declare const Buffer: typeof globalThis.Buffer;
declare const process: typeof globalThis.process;

declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      email: string;
      role: string;
      email_verified: boolean;
      user_metadata?: Record<string, unknown>;
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
        // Type assertion for Redis client compatibility
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

// Create and export a custom Request type that extends Express's Request
export type Request = ExpressRequest & {
  user?: {
    id: string;
    email: string;
    role: string;
    email_verified: boolean;
    user_metadata?: Record<string, unknown>;
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
export const apiLimiter = async (req: Request, res: Response, next: NextFunction) => {
  const { apiRateLimiter } = await getRateLimiters();
  return apiRateLimiter(req, res, next);
};

export const loginLimiter = async (req: Request, res: Response, next: NextFunction) => {
  const { authRateLimiter } = await getRateLimiters();
  return authRateLimiter(req, res, next);
};

export const refreshTokenLimiter = async (req: Request, res: Response, next: NextFunction) => {
  const { refreshTokenRateLimiter } = await getRateLimiters();
  return refreshTokenRateLimiter(req, res, next);
};

// Slow down for brute force protection
export const speedLimiter = slowDown({
  windowMs: 5 * 60 * 1000, // 5 minutes
  delayAfter: 3, // Allow 3 requests per 5 minutes, then...
  delayMs: 1000 // Add 1 second of delay per request above delayAfter
});

// Token validation schema (kept for potential future use)
// const tokenSchema = z.object({
//   sub: z.string().uuid(),
//   email: z.string().email(),
//   role: z.enum(['user', 'admin', 'moderator']).default('user'),
//   exp: z.number().int().positive(),
//   iat: z.number().int().positive()
// });

/**
 * Supabase Authentication Middleware with enhanced security and email verification
 */
export const authenticate = async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Check for token in Authorization header first
    let token: string | null = null;
    const authHeader = req.headers.authorization;
    
    // Also check for token in cookies (for web clients)
    if (!authHeader && req.cookies?.['sb-access-token']) {
      token = req.cookies['sb-access-token'];
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
        success: false,
        error: 'AUTH_REQUIRED',
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
          success: false,
          error: 'TOKEN_EXPIRED',
          message: 'Session expired. Please log in again.',
          code: 'TOKEN_EXPIRED'
        });
      }
      
      return res.status(401).json({
        success: false,
        error: 'INVALID_TOKEN',
        message: 'Invalid or expired token',
        code: 'INVALID_TOKEN'
      });
    }

    // Check if user exists and email is verified
    if (!user) {
      logger.warn('User not found during authentication');
      return res.status(401).json({
        success: false,
        error: 'USER_NOT_FOUND',
        message: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    // Check if email is verified
    if (!user.email_confirmed_at) {
      logger.warn('Email not verified', {
        userId: user.id,
        email: user.email,
        path: req.path
      });
      
      return res.status(403).json({
        success: false,
        error: 'EMAIL_NOT_VERIFIED',
        message: 'Please verify your email address to continue.',
        code: 'EMAIL_NOT_VERIFIED'
      });
    }

    // Attach user to request with minimal required data
    req.user = {
      id: user.id,
      email: user.email || '',
      role: user.user_metadata?.role || 'user',
      email_verified: !!user.email_confirmed_at,
      user_metadata: user.user_metadata || {}
    };
    
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

// CSRF token generation and validation
const CSRF_TOKEN_SECRET = process.env.CSRF_TOKEN_SECRET || 'dev_csrf_secret_for_development_only';
if (process.env.NODE_ENV !== 'development' && CSRF_TOKEN_SECRET.length < 32) {
  throw new Error('CSRF_TOKEN_SECRET must be at least 32 characters long');
}

const CSRF_TOKEN_AGE = 60 * 60 * 1000; // 1 hour (reduced from 24h)

/**
 * Generate a CSRF token with HMAC
 */
export const generateCsrfToken = (userId?: string): { token: string; cookie: string } => {
  const randomBytes = crypto.randomBytes(32);
  const timestamp = Date.now().toString();
  const hmac = crypto.createHmac('sha256', CSRF_TOKEN_SECRET);
  
  // Include user ID in the token if available for session binding
  const tokenData = userId ? `${userId}:${timestamp}` : timestamp;
  hmac.update(tokenData);
  
  const token = `${randomBytes.toString('hex')}.${hmac.digest('hex')}`;
  const expires = new Date(Date.now() + CSRF_TOKEN_AGE);
  
  // Enhanced cookie attributes
  const cookie = [
    `__Host-csrf_token=${token}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Strict',
    'Secure',
    'Partitioned',
    `Max-Age=${CSRF_TOKEN_AGE / 1000}`,
    `Expires=${expires.toUTCString()}`
  ].join('; ');
  
  return { token, cookie };
};

/**
 * CSRF Protection Middleware
 * Validates CSRF tokens using double-submit cookie pattern with HMAC verification
 */
export const csrfProtection = (req: Request, res: Response, next: NextFunction) => {
  // Skip CSRF check for safe HTTP methods
  if (['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(req.method)) {
    return next();
  }

  // Skip CSRF check for public endpoints that don't modify state
  const publicEndpoints = [
    '/auth/csrf-token',
    '/health',
    '/api/health'
  ];
  
  if (publicEndpoints.some(ep => req.path.endsWith(ep))) {
    return next();
  }

  // Get CSRF token from header and cookie
  const csrfToken = req.headers['x-csrf-token'] as string;
  const csrfCookie = req.cookies?.['__Host-csrf_token'];

  if (!csrfToken || !csrfCookie) {
    logger.warn('CSRF token missing', {
      path: req.path,
      method: req.method,
      hasToken: !!csrfToken,
      hasCookie: !!csrfCookie
    });
    
    return res.status(403).json({
      success: false,
      code: 'CSRF_TOKEN_REQUIRED',
      message: 'CSRF token is required for this request'
    });
  }

  // Verify tokens match (double-submit)
  if (csrfToken !== csrfCookie) {
    logger.warn('CSRF token mismatch', {
      path: req.path,
      method: req.method,
      tokenLength: csrfToken?.length,
      cookieLength: csrfCookie?.length
    });
    
    return res.status(403).json({
      success: false,
      code: 'INVALID_CSRF_TOKEN',
      message: 'Invalid CSRF token'
    });
  }

  // Verify token structure and HMAC
  try {
    const [randomPart, hmacValue] = csrfToken.split('.');
    if (!randomPart || !hmacValue || randomPart.length !== 64 || hmacValue.length !== 64) {
      throw new Error('Invalid token format');
    }

    // Recalculate HMAC
    const hmac = crypto.createHmac('sha256', CSRF_TOKEN_SECRET);
    hmac.update(randomPart);
    const calculatedHmac = hmac.digest('hex');

    if (!crypto.timingSafeEqual(
      Buffer.from(hmacValue, 'hex'),
      Buffer.from(calculatedHmac, 'hex')
    )) {
      throw new Error('Invalid token signature');
    }
  } catch (error) {
    logger.warn('CSRF token validation failed', {
      error: error instanceof Error ? error.message : 'Unknown error',
      path: req.path,
      method: req.method
    });
    
    return res.status(403).json({
      success: false,
      code: 'INVALID_CSRF_TOKEN',
      message: 'Invalid CSRF token format'
    });
  }

  // Token is valid, proceed
  next();
};

/**
 * Middleware to check if user's email is verified
 * Must be used after authenticate middleware
 */
export const requireVerifiedEmail = (req: Request, res: Response, next: NextFunction) => {
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

  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
  
  res.status(statusCode).json({
    success: false,
    code: err.name || 'INTERNAL_ERROR',
    message: process.env.NODE_ENV === 'development' 
      ? err.message 
      : 'An unexpected error occurred'
  });
};
