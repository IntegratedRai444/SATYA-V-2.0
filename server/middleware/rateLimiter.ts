import { Request, Response, NextFunction, RequestHandler } from 'express';
import { RateLimiterMemory, IRateLimiterOptions, RateLimiterRes } from 'rate-limiter-flexible';
import { errorResponse } from '../utils/apiResponse';
import { logger } from '../config/logger';
import type { AuthUser } from './auth';

// Extend Express Request type
declare global {
  namespace Express {
    interface Request {
      id?: string;
      ip?: string;
    }
  }
}

type RateLimiterType = 'general' | 'auth' | 'sensitive' | 'api';
type RateLimitScope = 'ip' | 'user' | 'global';

// Rate limiting options
const rateLimiterOptions: Record<RateLimiterType, IRateLimiterOptions> = {
  // General API rate limiting (1000 requests per 15 minutes per IP)
  general: {
    points: 1000, // Increased for general API use
    duration: 15 * 60, // Per 15 minutes
    blockDuration: 60 * 60, // Block for 1 hour if limit exceeded
    keyPrefix: 'rl_general',
  },
  
  // Authentication endpoints (5 requests per minute per IP)
  auth: {
    points: 5, // Reduced for auth endpoints
    duration: 60, // Per minute
    blockDuration: 15 * 60, // Block for 15 minutes
    keyPrefix: 'rl_auth',
  },
  
  // Sensitive operations (password reset, etc.)
  sensitive: {
    points: 3, // Very strict for sensitive operations
    duration: 60 * 60, // Per hour
    blockDuration: 6 * 60 * 60, // Block for 6 hours
    keyPrefix: 'rl_sensitive',
  },
  
  // API endpoints (higher limits for authenticated users)
  api: {
    points: 300, // Higher limit for API endpoints
    duration: 15 * 60, // Per 15 minutes
    blockDuration: 60 * 60, // Block for 1 hour
    keyPrefix: 'rl_api',
  },
};

// Create rate limiters for different scopes
const createLimiters = (type: RateLimiterType) => {
  const options = rateLimiterOptions[type];
  const points = options.points ?? 100; // Default to 100 points if undefined
  const baseConfig = {
    duration: options.duration,
    blockDuration: options.blockDuration,
    keyPrefix: options.keyPrefix,
  };

  return {
    ip: new RateLimiterMemory({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_ip`,
    }),
    user: new RateLimiterMemory({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_user`,
    }),
    global: new RateLimiterMemory({
      ...baseConfig,
      points: points * 5, // Higher limit for global
      keyPrefix: `${options.keyPrefix}_global`,
    }),
  };
};

// Initialize all limiters
const limiters = {
  general: createLimiters('general'),
  auth: createLimiters('auth'),
  sensitive: createLimiters('sensitive'),
  api: createLimiters('api'),
} as const;

// Get client IP address from request
const getClientIp = (req: Request): string => {
  // Check X-Forwarded-For header first (for proxies)
  const xForwardedFor = req.headers['x-forwarded-for'];
  if (typeof xForwardedFor === 'string') {
    return xForwardedFor.split(',')[0].trim();
  }
  
  // Fall back to other IP sources
  return req.ip || 
         (req.socket.remoteAddress ? req.socket.remoteAddress : 'unknown-ip');
};

// Rate limiter middleware with IP and user-based limiting
export const rateLimiter = (type: RateLimiterType = 'general'): RequestHandler => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const clientIp = getClientIp(req);
    const userId = req.user?.id?.toString() || 'anonymous';
    
    // Create rate limit keys
    const keys = {
      ip: `${type}:ip:${clientIp}`,
      user: `${type}:user:${userId}`,
      global: `${type}:global`
    };

    try {
      // Check all relevant rate limits
      const [ipRes, userRes, globalRes] = await Promise.all([
        limiters[type]?.ip.consume(keys.ip),
        limiters[type]?.user.consume(keys.user),
        limiters[type]?.global.consume(keys.global)
      ]);

      // Set rate limit headers (use the strictest limit)
      const remaining = Math.min(
        ipRes.remainingPoints,
        userRes.remainingPoints,
        globalRes.remainingPoints
      );
      
      const reset = Math.max(
        Math.ceil(ipRes.msBeforeNext / 1000),
        Math.ceil(userRes.msBeforeNext / 1000),
        Math.ceil(globalRes.msBeforeNext / 1000)
      );

      res.set({
        'X-RateLimit-Limit': rateLimiterOptions[type].points,
        'X-RateLimit-Remaining': remaining,
        'X-RateLimit-Reset': reset,
        'Retry-After': reset,
      });

      next();
    } catch (error) {
      const rateLimiterRes = error as RateLimiterRes;
      const retryAfter = Math.ceil(rateLimiterRes.msBeforeNext / 1000);
      
      res.set('Retry-After', String(retryAfter));
      
      logger.warn('Rate limit exceeded', {
        ip: clientIp,
        userId: userId !== 'anonymous' ? userId : undefined,
        path: req.path,
        method: req.method,
        type,
        retryAfter,
      });
      
      return errorResponse(
        res,
        {
          code: 'RATE_LIMIT_EXCEEDED',
          message: 'Too many requests, please try again later',
          retryAfter,
        },
        429
      );
    }
  };
};

// Specialized rate limiters
export const authRateLimiter = rateLimiter('auth');
export const sensitiveRateLimiter = rateLimiter('sensitive');
export const apiRateLimiter = rateLimiter('api');

// Apply rate limiting based on route
export const routeRateLimiter: RequestHandler = (req, res, next) => {
  // Apply different rate limits based on route
  if (req.path.startsWith('/api/auth')) {
    return authRateLimiter(req, res, next);
  }
  
  // Sensitive operations
  if (
    req.path.startsWith('/api/auth/password/reset') ||
    req.path.startsWith('/api/auth/verify-email') ||
    req.path.includes('change-password')
  ) {
    return sensitiveRateLimiter(req, res, next);
  }
  
  // API routes
  if (req.path.startsWith('/api/')) {
    return apiRateLimiter(req, res, next);
  }
  
  // Default rate limiter for all other routes
  return rateLimiter('general')(req, res, next);
};

// Export the default rate limiter
export default routeRateLimiter;

// Helper to get rate limit info for a specific key
export const getRateLimitInfo = async (type: RateLimiterType, scope: RateLimitScope, key: string) => {
  try {
    const limiter = limiters[type][scope];
    const res = await limiter.get(key);
    return res || null;
  } catch (error) {
    logger.error('Failed to get rate limit info:', error);
    return null;
  }
};

// Helper to reset rate limit for a specific key
export const resetRateLimit = async (type: RateLimiterType, scope: RateLimitScope, key: string) => {
  try {
    const limiter = limiters[type][scope];
    await limiter.delete(key);
    return true;
  } catch (error) {
    logger.error('Failed to reset rate limit:', error);
    return false;
  }
};
