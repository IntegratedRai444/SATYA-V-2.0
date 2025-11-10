import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import auditLogger from '../services/audit-logger';

interface RateLimitRule {
  windowMs: number;
  maxRequests: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  onLimitReached?: (req: Request, res: Response) => void;
}

interface RateLimitEntry {
  count: number;
  resetTime: number;
  firstRequest: number;
}

class AdvancedRateLimiter {
  private store: Map<string, RateLimitEntry> = new Map();
  private cleanupInterval: NodeJS.Timeout;

  constructor() {
    // Clean up expired entries every 5 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 5 * 60 * 1000);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.store.entries()) {
      if (now > entry.resetTime) {
        this.store.delete(key);
      }
    }
  }

  private getKey(req: Request, keyGenerator?: (req: Request) => string): string {
    if (keyGenerator) {
      return keyGenerator(req);
    }
    
    // Default key: IP + User ID (if authenticated)
    const ip = req.ip || req.connection.remoteAddress || 'unknown';
    const userId = (req as any).user?.user_id || 'anonymous';
    return `${ip}:${userId}`;
  }

  createMiddleware(rule: RateLimitRule) {
    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const key = this.getKey(req, rule.keyGenerator);
        const now = Date.now();
        
        let entry = this.store.get(key);
        
        if (!entry || now > entry.resetTime) {
          // Create new entry or reset expired entry
          entry = {
            count: 0,
            resetTime: now + rule.windowMs,
            firstRequest: now
          };
          this.store.set(key, entry);
        }

        // Check if request should be counted
        const shouldCount = !rule.skipSuccessfulRequests && !rule.skipFailedRequests;
        
        if (shouldCount) {
          entry.count++;
        }

        // Check if limit exceeded
        if (entry.count > rule.maxRequests) {
          // Log rate limit violation
          await auditLogger.logRateLimitViolation(
            req.ip || 'unknown',
            req.path,
            {
              method: req.method,
              userAgent: req.get('User-Agent'),
              count: entry.count,
              limit: rule.maxRequests,
              windowMs: rule.windowMs
            },
            req
          );

          // Call custom handler if provided
          if (rule.onLimitReached) {
            rule.onLimitReached(req, res);
            return;
          }

          // Default response
          const resetTime = Math.ceil((entry.resetTime - now) / 1000);
          
          res.status(429).json({
            error: 'Too Many Requests',
            message: 'Rate limit exceeded. Please try again later.',
            code: 'rate_limit_exceeded',
            retryAfter: resetTime,
            limit: rule.maxRequests,
            windowMs: rule.windowMs
          });
          
          return;
        }

        // Add rate limit headers
        const remaining = Math.max(0, rule.maxRequests - entry.count);
        const resetTime = Math.ceil((entry.resetTime - now) / 1000);
        
        res.set({
          'X-RateLimit-Limit': rule.maxRequests.toString(),
          'X-RateLimit-Remaining': remaining.toString(),
          'X-RateLimit-Reset': resetTime.toString(),
          'X-RateLimit-Window': rule.windowMs.toString()
        });

        next();
      } catch (error) {
        logger.error('Rate limiting error:', error);
        next(); // Continue on error to avoid blocking legitimate requests
      }
    };
  }

  // Get current stats for monitoring
  getStats(): {
    totalKeys: number;
    activeEntries: number;
    topConsumers: Array<{ key: string; count: number; resetTime: number }>;
  } {
    const now = Date.now();
    const activeEntries = Array.from(this.store.entries())
      .filter(([, entry]) => now <= entry.resetTime);
    
    const topConsumers = activeEntries
      .sort(([, a], [, b]) => b.count - a.count)
      .slice(0, 10)
      .map(([key, entry]) => ({
        key: key.replace(/:\d+$/, ':***'), // Mask user IDs for privacy
        count: entry.count,
        resetTime: entry.resetTime
      }));

    return {
      totalKeys: this.store.size,
      activeEntries: activeEntries.length,
      topConsumers
    };
  }

  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.store.clear();
  }
}

// Create singleton instance
const rateLimiter = new AdvancedRateLimiter();

// Predefined rate limiting rules
export const rateLimitRules = {
  // General API rate limiting
  api: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 100,
    keyGenerator: (req: Request) => {
      const ip = req.ip || 'unknown';
      const userId = (req as any).user?.user_id || 'anonymous';
      return `api:${ip}:${userId}`;
    }
  },

  // Authentication endpoints (stricter)
  auth: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 5,
    keyGenerator: (req: Request) => {
      const ip = req.ip || 'unknown';
      return `auth:${ip}`;
    },
    onLimitReached: async (req: Request, res: Response) => {
      await auditLogger.logSuspiciousActivity(
        undefined,
        'excessive_auth_attempts',
        {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          endpoint: req.path
        },
        'high',
        req
      );
    }
  },

  // Analysis endpoints (resource intensive)
  analysis: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 3,
    keyGenerator: (req: Request) => {
      const userId = (req as any).user?.user_id || 'anonymous';
      return `analysis:${userId}`;
    }
  },

  // File upload endpoints
  upload: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 5,
    keyGenerator: (req: Request) => {
      const userId = (req as any).user?.user_id || 'anonymous';
      return `upload:${userId}`;
    }
  },

  // WebSocket connections
  websocket: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 30, // 30 connection attempts per minute per IP
    skipSuccessfulRequests: false, // Count all connection attempts
    skipFailedRequests: false, // Count failed attempts too
    keyGenerator: (req: Request) => {
      const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
      return `ws:${ip}`;
    },
    onLimitReached: (req: Request, res: Response) => {
      const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
      logger.warn(`WebSocket rate limit reached for IP: ${ip}`);
      
      if (!res.headersSent) {
        res.status(429).json({
          success: false,
          error: 'Too many connection attempts. Please try again later.'
        });
      }
    }
  }
};

// Create middleware functions
export const apiRateLimit = rateLimiter.createMiddleware(rateLimitRules.api);
export const authRateLimit = rateLimiter.createMiddleware(rateLimitRules.auth);
export const analysisRateLimit = rateLimiter.createMiddleware(rateLimitRules.analysis);
export const uploadRateLimit = rateLimiter.createMiddleware(rateLimitRules.upload);
export const websocketRateLimit = rateLimiter.createMiddleware(rateLimitRules.websocket);

// Export rate limiter instance for stats
export { rateLimiter };

export default rateLimiter;