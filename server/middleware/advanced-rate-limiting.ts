import type { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import auditLogger from '../services/audit-logger';

interface RateLimitRule {
  windowMs: number;
  maxRequests: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  onLimitReached?: (req: Request, res: Response) => void;
  blockDuration?: number; // Duration to block after limit is reached (ms)
  trustProxy?: boolean; // Whether to trust X-Forwarded-For header
  message?: string; // Custom rate limit message
  statusCode?: number; // Custom status code for rate limited responses
}

interface RateLimitEntry {
  count: number;
  resetTime: number;
  firstRequest: number;
}

class AdvancedRateLimiter {
  private store: Map<string, RateLimitEntry> = new Map();
  private blockedIPs: Map<string, number> = new Map();
  private cleanupInterval: NodeJS.Timeout;
  private blockCleanupInterval: NodeJS.Timeout;

  constructor() {
    // Clean up expired rate limit entries every 5 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 5 * 60 * 1000);

    // Clean up blocked IPs every minute
    this.blockCleanupInterval = setInterval(() => {
      this.cleanupBlockedIPs();
    }, 60 * 1000);
  }

  private cleanup(): void {
    const now = Date.now();
    // Clean up expired rate limit entries
    const expiredKeys: string[] = [];
    for (const [key, entry] of this.store.entries()) {
      if (now > entry.resetTime) {
        expiredKeys.push(key);
      }
    }
    expiredKeys.forEach(key => this.store.delete(key));
  }

  private cleanupBlockedIPs(): void {
    const now = Date.now();
    // Clean up expired blocked IPs
    for (const [ip, expiry] of this.blockedIPs.entries()) {
      if (now > expiry) {
        this.blockedIPs.delete(ip);
      }
    }
  }

  private isIPBlocked(ip: string): boolean {
    const expiry = this.blockedIPs.get(ip);
    if (!expiry) return false;
    if (Date.now() > expiry) {
      this.blockedIPs.delete(ip);
      return false;
    }
    return true;
  }

  private getIP(req: Request): string {
    // Get client IP, considering proxy headers if trustProxy is true
    const xForwardedFor = req.headers['x-forwarded-for'];
    const xForwardedForIp = Array.isArray(xForwardedFor) 
      ? xForwardedFor[0]?.trim()
      : typeof xForwardedFor === 'string'
        ? xForwardedFor.split(',')[0]?.trim()
        : null;
        
    const ip = xForwardedForIp || 
              req.socket.remoteAddress || 
              'unknown';
    
    // Basic IP validation and normalization
    if (ip === '::1') return '127.0.0.1';
    if (ip.startsWith('::ffff:')) return ip.substring(7);
    return ip;
  }

  private getKey(req: Request, rule: RateLimitRule): string {
    // Check for custom key generator first
    if (rule.keyGenerator) {
      return rule.keyGenerator(req);
    }
    
    // Default key: IP + User ID (if authenticated) + path
    const ip = this.getIP(req);
    const userId = (req as any).user?.user_id || 'anonymous';
    const path = req.path;
    
    return `${ip}:${userId}:${path}`;
  }

  createMiddleware(rule: RateLimitRule) {
    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const ip = this.getIP(req);
        
        // Check if IP is blocked
        if (this.isIPBlocked(ip)) {
          const retryAfter = Math.ceil((this.blockedIPs.get(ip)! - Date.now()) / 1000);
          res.set('Retry-After', String(retryAfter));
          return res.status(429).json({
            success: false,
            error: 'Too many requests',
            message: rule.message || 'Rate limit exceeded. Please try again later.',
            retryAfter,
            code: 'RATE_LIMIT_EXCEEDED'
          });
        }
        
        const key = this.getKey(req, rule);
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
        const shouldCount = !rule.skipSuccessfulRequests || 
                          (rule.skipSuccessfulRequests && !(res.statusCode >= 200 && res.statusCode < 300)) ||
                          (rule.skipFailedRequests && res.statusCode >= 400);
        
        if (shouldCount) {
          entry.count++;
        }

        // Calculate rate limit values
        const remaining = Math.max(0, rule.maxRequests - entry.count);
        const resetTime = Math.ceil((entry.resetTime - now) / 1000);
        
        // Set rate limit headers
        const headers: Record<string, string> = {
          'X-RateLimit-Limit': rule.maxRequests.toString(),
          'X-RateLimit-Remaining': remaining.toString(),
          'X-RateLimit-Reset': entry.resetTime.toString()
        };
        
        // Only set Retry-After if we're close to the limit
        if (remaining < Math.ceil(rule.maxRequests * 0.2)) { // 20% of limit remaining
          headers['Retry-After'] = resetTime.toString();
        }
        
        res.set(headers);

        // Check if limit exceeded
        if (entry.count > rule.maxRequests) {
          // Block IP if blockDuration is set
          if (rule.blockDuration) {
            this.blockedIPs.set(ip, now + rule.blockDuration);
          }
          // Log rate limit violation
          const clientInfo = {
            ip,
            method: req.method,
            path: req.path,
            userAgent: req.get('User-Agent'),
            userId: (req as any).user?.user_id,
            count: entry.count,
            limit: rule.maxRequests,
            windowMs: rule.windowMs,
            resetTime: entry.resetTime,
            retryAfter: Math.ceil((entry.resetTime - now) / 1000)
          };

          logger.warn('Rate limit exceeded', clientInfo);
          
          try {
            await auditLogger.logRateLimitViolation(
              ip,
              req.path,
              clientInfo,
              req
            );
          } catch (logError) {
            logger.error('Failed to log rate limit violation', { error: logError });
          }

          // Call custom handler if provided
          if (rule.onLimitReached) {
            try {
              rule.onLimitReached(req, res);
            } catch (handlerError) {
              logger.error('Error in rate limit handler', { error: handlerError });
            }
            return;
          }

          // Default response
          res.status(rule.statusCode || 429).json({
            success: false,
            error: 'Too Many Requests',
            message: rule.message || 'Rate limit exceeded. Please try again later.',
            code: 'RATE_LIMIT_EXCEEDED',
            retryAfter: clientInfo.retryAfter,
            limit: rule.maxRequests,
            windowMs: rule.windowMs,
            ...(rule.blockDuration ? { blockDuration: rule.blockDuration } : {})
          });
          
          return;
        }

        // Rate limit headers already set above
        // No need to set them again
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
    if (this.blockCleanupInterval) {
      clearInterval(this.blockCleanupInterval);
    }
    this.store.clear();
    this.blockedIPs.clear();
  }
}

// Create singleton instance
const rateLimiter = new AdvancedRateLimiter();

// Predefined rate limiting rules
export const rateLimitRules = {
  // Default API rate limiting
  api: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 100,
    skipFailedRequests: true,
    message: 'Too many requests from this IP, please try again after 15 minutes',
    blockDuration: 15 * 60 * 1000, // 15 minutes block
    statusCode: 429
  },
  // Authentication endpoints (stricter)
  auth: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 5,
    message: 'Too many login attempts. Please try again later.',
    blockDuration: 30 * 60 * 1000, // 30 minutes block for repeated auth failures
    statusCode: 429,
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
    maxRequests: 10,
    message: 'Too many analysis requests. Please wait before trying again.',
    blockDuration: 5 * 60 * 1000, // 5 minutes block
    statusCode: 429,
    keyGenerator: (req: Request) => {
      const userId = (req as any).user?.user_id || 'anonymous';
      return `analysis:${userId}`;
    }
  },
  // File upload endpoints
  upload: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 5,
    message: 'Too many uploads. Please wait before trying again.',
    blockDuration: 15 * 60 * 1000, // 15 minutes block
    statusCode: 429,
    keyGenerator: (req: Request) => {
      const userId = (req as any).user?.user_id || 'anonymous';
      return `upload:${userId}`;
    }
  },

  // WebSocket connections
  websocket: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 30, // 30 connection attempts per minute per IP
    message: 'Too many connection attempts. Please try again later.',
    blockDuration: 5 * 60 * 1000, // 5 minutes block
    statusCode: 429,
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