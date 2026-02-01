import { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { logger } from '../../config/logger';

/**
 * Production Security Configuration
 * 
 * This module provides enhanced security configurations specifically for production
 * environments, implementing defense-in-depth security strategies.
 */

export interface ProductionSecurityConfig {
  enableCSP: boolean;
  enableHSTS: boolean;
  enableRateLimiting: boolean;
  enableIPWhitelist: boolean;
  allowedIPs: string[];
  customSecurityHeaders: Record<string, string>;
}

const defaultProductionConfig: ProductionSecurityConfig = {
  enableCSP: true,
  enableHSTS: true,
  enableRateLimiting: true,
  enableIPWhitelist: false,
  allowedIPs: [],
  customSecurityHeaders: {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
  },
};

/**
 * Content Security Policy for production
 */
const productionCSP = {
  directives: {
    defaultSrc: ["'self'"],
    styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
    fontSrc: ["'self'", 'https://fonts.gstatic.com'],
    imgSrc: ["'self'", 'data:', 'https:'],
    scriptSrc: ["'self'"],
    connectSrc: ["'self'", 'wss:', 'https:'],
    frameSrc: ["'none'"],
    objectSrc: ["'none'"],
    mediaSrc: ["'self'"],
    manifestSrc: ["'self'"],
    workerSrc: ["'self'"],
  },
};

/**
 * Enhanced rate limiting for production
 */
const productionRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes',
  },
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req: Request, res: Response) => {
    logger.warn('Rate limit exceeded', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      path: req.path,
      method: req.method,
    });
    
    res.status(429).json({
      success: false,
      error: 'Rate limit exceeded',
      message: 'Too many requests from this IP, please try again later.',
      retryAfter: '15 minutes',
    });
  },
});

/**
 * Strict rate limiting for sensitive endpoints
 */
const strictRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // Very strict for sensitive operations
  message: {
    error: 'Too many sensitive requests from this IP',
    retryAfter: '15 minutes',
  },
  skipSuccessfulRequests: false,
  skipFailedRequests: false,
});

/**
 * IP Whitelist middleware
 */
const ipWhitelist = (allowedIPs: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const clientIP = req.ip || req.connection.remoteAddress;
    
    if (!allowedIPs.includes(clientIP as string)) {
      logger.warn('Unauthorized IP access attempt', {
        ip: clientIP,
        userAgent: req.get('User-Agent'),
        path: req.path,
      });
      
      return res.status(403).json({
        success: false,
        error: 'Access denied',
        message: 'Your IP is not authorized to access this resource',
      });
    }
    
    next();
  };
};

/**
 * Security headers middleware
 */
const securityHeaders = (config: ProductionSecurityConfig) => {
  return (req: Request, res: Response, next: NextFunction) => {
    // Apply custom security headers
    Object.entries(config.customSecurityHeaders).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
    
    // Remove server information
    res.removeHeader('Server');
    res.removeHeader('X-Powered-By');
    
    next();
  };
};

/**
 * Request validation middleware
 */
const validateRequest = (req: Request, res: Response, next: NextFunction) => {
  // Check for suspicious patterns
  const suspiciousPatterns = [
    /\.\./,  // Path traversal
    /<script/i,  // XSS attempts
    /union.*select/i,  // SQL injection attempts
    /javascript:/i,  // JavaScript protocol
  ];
  
  const url = req.url;
  const userAgent = req.get('User-Agent') || '';
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(url) || pattern.test(userAgent)) {
      logger.warn('Suspicious request detected', {
        ip: req.ip,
        url,
        userAgent,
        pattern: pattern.source,
      });
      
      return res.status(400).json({
        success: false,
        error: 'Invalid request',
        message: 'Your request contains suspicious content',
      });
    }
  }
  
  next();
};

/**
 * Apply production security middleware
 */
export const applyProductionSecurity = (
  app: any,
  config: Partial<ProductionSecurityConfig> = {}
) => {
  const securityConfig = { ...defaultProductionConfig, ...config };
  
  // Apply Helmet with custom CSP
  if (securityConfig.enableCSP) {
    app.use(helmet({
      contentSecurityPolicy: productionCSP,
      hsts: securityConfig.enableHSTS ? {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true,
      } : false,
    }));
  }
  
  // Apply rate limiting
  if (securityConfig.enableRateLimiting) {
    app.use('/api/', productionRateLimit);
    
    // Strict rate limiting for sensitive endpoints
    app.use('/api/v2/auth/', strictRateLimit);
    app.use('/api/v2/user/', strictRateLimit);
    app.use('/api/v2/analysis/', strictRateLimit);
  }
  
  // Apply IP whitelist if enabled
  if (securityConfig.enableIPWhitelist && securityConfig.allowedIPs.length > 0) {
    app.use('/api/', ipWhitelist(securityConfig.allowedIPs));
  }
  
  // Apply security headers
  app.use(securityHeaders(securityConfig));
  
  // Apply request validation
  app.use('/api/', validateRequest);
  
  logger.info('Production security middleware applied', {
    csp: securityConfig.enableCSP,
    hsts: securityConfig.enableHSTS,
    rateLimiting: securityConfig.enableRateLimiting,
    ipWhitelist: securityConfig.enableIPWhitelist,
  });
  
  return app;
};

/**
 * Security audit middleware
 */
export const securityAudit = (req: Request, res: Response, next: NextFunction) => {
  const auditData = {
    timestamp: new Date().toISOString(),
    ip: req.ip,
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent'),
    contentType: req.get('Content-Type'),
    contentLength: req.get('Content-Length'),
  };
  
  // Log security-relevant requests
  if (req.path.startsWith('/api/v2/auth/') || 
      req.path.startsWith('/api/v2/user/') ||
      req.path.startsWith('/api/v2/analysis/')) {
    logger.info('Security audit', auditData);
  }
  
  next();
};

export default {
  applyProductionSecurity,
  securityAudit,
  productionRateLimit,
  strictRateLimit,
  defaultProductionConfig,
};
