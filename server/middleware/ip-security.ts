import type { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

interface IPSecurityConfig {
  whitelist: string[];
  blacklist: string[];
  enabled: boolean;
  blockUnknown: boolean;
}

/**
 * IP-based Security Middleware
 * Implements IP whitelist/blacklist functionality
 */
export const ipSecurity = (config: Partial<IPSecurityConfig> = {}) => {
  const securityConfig: IPSecurityConfig = {
    whitelist: process.env.IP_WHITELIST?.split(',').map(ip => ip.trim()) || [],
    blacklist: process.env.IP_BLACKLIST?.split(',').map(ip => ip.trim()) || [],
    enabled: process.env.IP_SECURITY_ENABLED === 'true',
    blockUnknown: process.env.IP_BLOCK_UNKNOWN === 'true',
    ...config
  };

  return (req: Request, res: Response, next: NextFunction) => {
    if (!securityConfig.enabled) {
      return next();
    }

    const clientIP = getClientIP(req);
    
    // Check blacklist first
    if (securityConfig.blacklist.length > 0) {
      const isBlacklisted = securityConfig.blacklist.some(ip => 
        clientIP === ip || clientIP.startsWith(ip + '.')
      );
      
      if (isBlacklisted) {
        logger.warn('Blocked blacklisted IP', { ip: clientIP, path: req.path });
        return res.status(403).json({
          error: 'IP_BLOCKED',
          message: 'Access denied from this IP address'
        });
      }
    }

    // Check whitelist if enabled
    if (securityConfig.whitelist.length > 0) {
      const isWhitelisted = securityConfig.whitelist.some(ip => 
        clientIP === ip || clientIP.startsWith(ip + '.')
      );
      
      if (!isWhitelisted) {
        logger.warn('Blocked non-whitelisted IP', { ip: clientIP, path: req.path });
        return res.status(403).json({
          error: 'IP_NOT_ALLOWED',
          message: 'Access denied from this IP address'
        });
      }
    }

    // Block unknown IPs if enabled
    if (securityConfig.blockUnknown && securityConfig.whitelist.length === 0) {
      const isPrivate = isPrivateIP(clientIP);
      if (!isPrivate) {
        logger.warn('Blocked unknown IP', { ip: clientIP, path: req.path });
        return res.status(403).json({
          error: 'IP_UNKNOWN',
          message: 'Access denied from unknown IP address'
        });
      }
    }

    next();
  };
};

/**
 * Get client IP address from request
 */
function getClientIP(req: Request): string {
  // Check X-Forwarded-For header first (for proxies)
  const xForwardedFor = req.headers['x-forwarded-for'];
  if (typeof xForwardedFor === 'string') {
    return xForwardedFor.split(',')[0].trim();
  }
  
  // Check X-Real-IP header
  const xRealIP = req.headers['x-real-ip'];
  if (typeof xRealIP === 'string') {
    return xRealIP.trim();
  }
  
  // Fall back to other IP sources
  return req.ip || req.socket?.remoteAddress || 'unknown-ip';
}

/**
 * Check if IP is private/local
 */
function isPrivateIP(ip: string): boolean {
  // IPv4 private ranges
  const privateRanges = [
    /^10\./,          // 10.0.0.0/8
    /^172\.(1[6-9]|2[0-9]|3[0-1])\./,  // 172.16.0.0/12
    /^192\.168\./,    // 192.168.0.0/16
    /^127\./,         // 127.0.0.0/8
    /^169\.254\./,    // 169.254.0.0/16
    /^::1$/,          // IPv6 localhost
    /^fc00:/,         // IPv6 unique local
    /^fe80:/,         // IPv6 link-local
  ];
  
  return privateRanges.some(range => range.test(ip));
}

/**
 * Rate limiting by IP with stricter rules for unknown IPs
 */
export const strictRateLimit = (req: Request, res: Response, next: NextFunction) => {
  const clientIP = getClientIP(req);
  const isPrivate = isPrivateIP(clientIP);
  
  // Apply stricter rate limiting for public IPs
  if (!isPrivate) {
    // Add custom header for stricter rate limiting
    req.headers['x-strict-rate-limit'] = 'true';
  }
  
  next();
};

/**
 * Geographic IP blocking middleware
 */
export const geoBlock = (allowedCountries: string[] = ['US', 'CA', 'GB', 'AU']) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const clientIP = getClientIP(req);
    
    try {
      // Skip geo-blocking for private IPs
      if (isPrivateIP(clientIP)) {
        return next();
      }
      
      // TODO: Integrate with IP geolocation service
      // For now, allow all requests
      next();
    } catch (error) {
      logger.error('Geo-blocking error:', error);
      next(); // Fail open
    }
  };
};

export default {
  ipSecurity,
  strictRateLimit,
  geoBlock
};
