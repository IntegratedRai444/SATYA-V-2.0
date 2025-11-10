import { Request, Response, NextFunction } from 'express';
import { defaultSecurityConfig as securityConfig } from './security';
import { logger } from './logger';
import { validationResult } from 'express-validator';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import cors from 'cors';

export const configureSecurity = (app: any) => {
  // 1. Set security HTTP headers using Helmet
  if (securityConfig.securityHeaders.enabled) {
    app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          ...securityConfig.securityHeaders.contentSecurityPolicy.directives,
        },
      },
      hsts: securityConfig.securityHeaders.hsts,
      frameguard: { action: 'deny' },
      noSniff: true,
      xssFilter: true,
    }));
  }

  // 2. Enable CORS
  if (securityConfig.cors.enabled) {
    app.use(cors({
      origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
        if (!origin || securityConfig.cors.allowedOrigins.includes(origin) || 
            securityConfig.cors.allowedOrigins.includes('*')) {
          callback(null, true);
        } else {
          logger.warn(`CORS blocked request from origin: ${origin}`);
          callback(new Error('Not allowed by CORS'));
        }
      },
      credentials: securityConfig.cors.credentials,
      methods: securityConfig.cors.allowedMethods,
      allowedHeaders: securityConfig.cors.allowedHeaders,
      exposedHeaders: securityConfig.cors.exposedHeaders,
      maxAge: securityConfig.cors.maxAge
    }));
  }

  // 3. Rate limiting
  if (securityConfig.rateLimiting.enabled) {
    const limiter = rateLimit({
      windowMs: securityConfig.rateLimiting.default.windowMs,
      max: securityConfig.rateLimiting.default.maxRequests,
      standardHeaders: true,
      legacyHeaders: false,
      message: 'Too many requests, please try again later.',
      skip: (req) => {
        // Skip rate limiting for certain paths
        const skipPaths = ['/health', '/metrics', '/favicon.ico'];
        return skipPaths.some(path => req.path.startsWith(path));
      },
      handler: (req, res) => {
        logger.warn(`Rate limit exceeded for IP: ${req.ip}`);
        res.status(429).json({
          success: false,
          message: 'Too many requests, please try again later.',
          retryAfter: (req as any).rateLimit?.resetTime || 60
        });
      }
    });

    app.use(limiter);
  }

  // 4. Request validation middleware
  app.use((req: Request, res: Response, next: NextFunction) => {
    // Log request details for security monitoring
    logger.info(`[${new Date().toISOString()}] ${req.method} ${req.path}`, {
      ip: req.ip,
      userAgent: req.get('user-agent'),
      contentType: req.get('content-type'),
      contentLength: req.get('content-length')
    });
    next();
  });

  // 5. Security headers
  app.use((req: Request, res: Response, next: NextFunction) => {
    // Set security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', securityConfig.headers.referrerPolicy);
    res.setHeader('Permissions-Policy', 
Object.entries(securityConfig.headers.permissionsPolicy)
        .map(([key, value]) => {
          const values = Array.isArray(value) ? value : [value];
          return `${key}=(${values.join(' ')})`;
        })
        .join(', ')
    );
    next();
  });

  // 6. Error handling for security-related issues
  app.use((err: any, req: Request, res: Response, next: NextFunction) => {
    if (err.name === 'UnauthorizedError') {
      logger.warn(`Unauthorized access attempt: ${err.message}`, {
        ip: req.ip,
        path: req.path,
        method: req.method
      });
      return res.status(401).json({
        success: false,
        message: 'Unauthorized access',
        code: 'UNAUTHORIZED'
      });
    }
    next(err);
  });

  logger.info('Security middleware configured successfully');
};

// Export security configuration
export const getSecurityConfig = () => securityConfig;
