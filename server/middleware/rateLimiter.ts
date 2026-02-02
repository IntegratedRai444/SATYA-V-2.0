import type { RequestHandler, Request } from 'express';
import { RateLimiterMemory, RateLimiterRedis, IRateLimiterOptions, RateLimiterRes } from 'rate-limiter-flexible';
import { createClient } from 'redis';
import { errorResponse } from '../utils/apiResponse';
import { logger } from '../config/logger';

type RateLimiterType = 'general' | 'auth' | 'sensitive' | 'api';
type RateLimitScope = 'ip' | 'user' | 'global';

// Rate limiting options with environment-based configuration
const getEnvNumber = (key: string, defaultValue: number): number => {
  const value = process.env[key];
  return value ? parseInt(value, 10) : defaultValue;
};

const rateLimiterOptions: Record<RateLimiterType, IRateLimiterOptions> = {
  // General API rate limiting
  general: {
    points: getEnvNumber('RATE_LIMIT_GENERAL_POINTS', 1000),
    duration: getEnvNumber('RATE_LIMIT_GENERAL_DURATION', 15 * 60),
    blockDuration: getEnvNumber('RATE_LIMIT_GENERAL_BLOCK', 60 * 60),
    keyPrefix: 'rl_general',
  },
  
  // Authentication endpoints
  auth: {
    points: getEnvNumber('RATE_LIMIT_AUTH_POINTS', 5),
    duration: getEnvNumber('RATE_LIMIT_AUTH_DURATION', 60),
    blockDuration: getEnvNumber('RATE_LIMIT_AUTH_BLOCK', 15 * 60),
    keyPrefix: 'rl_auth',
  },
  
  // Sensitive operations
  sensitive: {
    points: getEnvNumber('RATE_LIMIT_SENSITIVE_POINTS', 3),
    duration: getEnvNumber('RATE_LIMIT_SENSITIVE_DURATION', 60 * 60),
    blockDuration: getEnvNumber('RATE_LIMIT_SENSITIVE_BLOCK', 6 * 60 * 60),
    keyPrefix: 'rl_sensitive',
  },
  
  // API endpoints
  api: {
    points: getEnvNumber('RATE_LIMIT_API_POINTS', 300),
    duration: getEnvNumber('RATE_LIMIT_API_DURATION', 15 * 60),
    blockDuration: getEnvNumber('RATE_LIMIT_API_BLOCK', 60 * 60),
    keyPrefix: 'rl_api',
  },
};

// Redis client for distributed rate limiting
let redisClient: ReturnType<typeof createClient> | null = null;

// Initialize Redis client
const initializeRedis = async () => {
  if (process.env.REDIS_URL && !redisClient) {
    try {
      redisClient = createClient({
        url: process.env.REDIS_URL,
        socket: {
          reconnectStrategy: (retries) => Math.min(retries * 100, 5000)
        }
      });
      
      await redisClient.connect();
      logger.info('Redis client connected for rate limiting');
    } catch (error) {
      logger.error('Failed to connect Redis for rate limiting:', error);
      redisClient = null;
    }
  }
};

// Initialize Redis on startup
initializeRedis();

// Create rate limiters for different scopes
const createLimiters = (type: RateLimiterType) => {
  const options = rateLimiterOptions[type];
  const points = options.points ?? 100;
  const baseConfig = {
    duration: options.duration,
    blockDuration: options.blockDuration,
    keyPrefix: options.keyPrefix,
  };

  // Use Redis if available, otherwise fall back to memory
  const store = redisClient ? new RateLimiterRedis({
    redis: redisClient,
    sendCommand: (...args: unknown[]) => (redisClient as any).sendCommand(...args),
  }) : null;

  return {
    ip: store ? new RateLimiterRedis({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_ip`,
      redis: redisClient,
      sendCommand: (...args: unknown[]) => (redisClient as any).sendCommand(...args),
    }) : new RateLimiterMemory({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_ip`,
    }),
    user: store ? new RateLimiterRedis({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_user`,
      redis: redisClient,
      sendCommand: (...args: unknown[]) => (redisClient as any).sendCommand(...args),
    }) : new RateLimiterMemory({
      ...baseConfig,
      points,
      keyPrefix: `${options.keyPrefix}_user`,
    }),
    global: store ? new RateLimiterRedis({
      ...baseConfig,
      points: points * 5,
      keyPrefix: `${options.keyPrefix}_global`,
      redis: redisClient,
      sendCommand: (...args: unknown[]) => (redisClient as any).sendCommand(...args),
    }) : new RateLimiterMemory({
      ...baseConfig,
      points: points * 5,
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
         (req.socket?.remoteAddress ? req.socket.remoteAddress : 'unknown-ip');
};

// Rate limiter middleware with IP and user-based limiting
export const rateLimiter = (type: RateLimiterType = 'general'): RequestHandler => {
  return async (req, res, next) => {
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
