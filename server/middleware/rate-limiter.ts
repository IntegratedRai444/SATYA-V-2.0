import { setInterval } from 'timers';
import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

interface RateLimitEntry {
  count: number;
  resetTime: number;
  lastAccess: number;
}

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  maxConcurrent: number;
  message: string;
}

class RateLimiter {
  private userLimits = new Map<string, RateLimitEntry>();
  private concurrentJobs = new Map<string, number>();
  private configs = new Map<string, RateLimitConfig>();

  constructor() {
    // Configure different limits for different endpoint types
    this.configs.set('analysis', {
      windowMs: 60 * 1000, // 1 minute
      maxRequests: 100, // 100 analyses per minute (increased for testing)
      maxConcurrent: 10, // 10 concurrent jobs (increased for testing)
      message: 'Too many analyses. Please wait before trying again.'
    });

    this.configs.set('upload', {
      windowMs: 60 * 1000, // 1 minute  
      maxRequests: 50, // 50 uploads per minute (increased for testing)
      maxConcurrent: 10, // 10 concurrent uploads (increased for testing)
      message: 'Too many file uploads. Please wait before trying again.'
    });

    // Cleanup expired entries every 5 minutes
    setInterval(() => this.cleanup(), 5 * 60 * 1000);
  }

  // Middleware function
  middleware(type: 'analysis' | 'upload' = 'analysis') {
    const config = this.configs.get(type);
    if (!config) {
      throw new Error(`Unknown rate limit type: ${type}`);
    }

    return (req: Request, res: Response, next: NextFunction) => {
      const userId = (req as any).user?.id;
      
      // Skip rate limiting for unauthenticated requests (they'll be caught by auth middleware)
      if (!userId) {
        return next();
      }

      const now = Date.now();
      const userKey = `${type}:${userId}`;

      // Check rate limit
      const userLimit = this.userLimits.get(userKey);
      
      if (!userLimit || now > userLimit.resetTime) {
        // New window or expired window
        this.userLimits.set(userKey, {
          count: 1,
          resetTime: now + config.windowMs,
          lastAccess: now
        });
      } else {
        // Existing window
        userLimit.count++;
        userLimit.lastAccess = now;
        
        if (userLimit.count > config.maxRequests) {
          logger.warn('[RATE LIMIT] User exceeded rate limit', {
            userId,
            type,
            count: userLimit.count,
            limit: config.maxRequests,
            windowMs: config.windowMs
          });

          return res.status(429).json({
            success: false,
            error: {
              code: 'RATE_LIMIT_EXCEEDED',
              message: config.message,
              retryAfter: Math.ceil((userLimit.resetTime - now) / 1000)
            }
          });
        }
      }

      // Check concurrent job limit
      const currentConcurrent = this.concurrentJobs.get(userId) || 0;
      
      if (currentConcurrent >= config.maxConcurrent) {
        logger.warn('[RATE LIMIT] User exceeded concurrent job limit', {
          userId,
          type,
          currentConcurrent,
          limit: config.maxConcurrent
        });

        return res.status(429).json({
          success: false,
          error: {
            code: 'CONCURRENT_JOB_LIMIT_EXCEEDED',
            message: 'Too many active analyses. Please wait for current jobs to complete.',
            retryAfter: 60 // Suggest checking back in 1 minute
          }
        });
      }

      // Add rate limit headers
      res.set({
        'X-RateLimit-Limit': config.maxRequests.toString(),
        'X-RateLimit-Remaining': Math.max(0, config.maxRequests - (userLimit?.count || 0)).toString(),
        'X-RateLimit-Reset': new Date((userLimit?.resetTime || now) + config.windowMs).toISOString(),
        'X-Concurrent-Limit': config.maxConcurrent.toString(),
        'X-Concurrent-Remaining': Math.max(0, config.maxConcurrent - currentConcurrent).toString()
      });

      next();
    };
  }

  // Increment concurrent jobs for user
  incrementConcurrent(userId: string): void {
    const current = this.concurrentJobs.get(userId) || 0;
    this.concurrentJobs.set(userId, current + 1);
  }

  // Decrement concurrent jobs for user
  decrementConcurrent(userId: string): void {
    const current = this.concurrentJobs.get(userId) || 0;
    this.concurrentJobs.set(userId, Math.max(0, current - 1));
  }

  // Get current status for a user
  getUserStatus(userId: string, type: 'analysis' | 'upload' = 'analysis') {
    const userKey = `${type}:${userId}`;
    const limit = this.userLimits.get(userKey);
    const concurrent = this.concurrentJobs.get(userId) || 0;
    const config = this.configs.get(type);

    if (!limit || !config) {
      return {
        withinLimit: true,
        remaining: config?.maxRequests || 0,
        concurrentRemaining: config?.maxConcurrent || 0,
        resetTime: new Date()
      };
    }

    const now = Date.now();
    const resetTime = limit && limit.resetTime > now ? new Date(limit.resetTime) : new Date(now + config.windowMs);
    const remaining = limit ? Math.max(0, config.maxRequests - limit.count) : config.maxRequests;
    const concurrentRemaining = Math.max(0, config.maxConcurrent - concurrent);

    return {
      withinLimit: remaining > 0 && concurrent < config.maxConcurrent,
      remaining,
      concurrentRemaining,
      resetTime
    };
  }

  // Cleanup expired entries
  private cleanup(): void {
    const now = Date.now();
    
    // Clean rate limit entries
    for (const [key, entry] of this.userLimits.entries()) {
      if (now > entry.resetTime) {
        this.userLimits.delete(key);
      }
    }

    // Clean concurrent job entries (remove users with 0 concurrent jobs)
    for (const [userId, count] of this.concurrentJobs.entries()) {
      if (count <= 0) {
        this.concurrentJobs.delete(userId);
      }
    }

    logger.debug('[RATE LIMIT] Cleanup completed', {
      rateLimitEntries: this.userLimits.size,
      concurrentEntries: this.concurrentJobs.size
    });
  }

  // Get statistics
  getStats() {
    return {
      totalUsers: this.userLimits.size,
      activeUsers: this.concurrentJobs.size,
      configs: Array.from(this.configs.entries()).map(([type, config]) => ({
        type,
        maxRequests: config.maxRequests,
        maxConcurrent: config.maxConcurrent,
        windowMs: config.windowMs
      }))
    };
  }
}

// Singleton instance
const rateLimiter = new RateLimiter();
export default rateLimiter;

// Convenience middleware functions
export const analysisRateLimit = rateLimiter.middleware('analysis');
export const uploadRateLimit = rateLimiter.middleware('upload');

// Helper functions for job tracking
export const incrementConcurrentJobs = (userId: string) => rateLimiter.incrementConcurrent(userId);
export const decrementConcurrentJobs = (userId: string) => rateLimiter.decrementConcurrent(userId);
