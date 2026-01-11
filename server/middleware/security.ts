import type { Request, Response, NextFunction, RequestHandler } from 'express';
import { createHmac, timingSafeEqual, randomBytes } from 'crypto';
import { logger } from '../config/logger';
import { ApiKeyService } from '../services/apiKeyService';
import { AuthUser } from './auth';
import { validationResult } from 'express-validator';
import { rateLimit } from 'express-rate-limit';
import helmet, { contentSecurityPolicy, crossOriginEmbedderPolicy, crossOriginOpenerPolicy, crossOriginResourcePolicy, dnsPrefetchControl, frameguard, hidePoweredBy, hsts, ieNoOpen, noSniff, originAgentCluster, referrerPolicy, xssFilter } from 'helmet';
// @ts-ignore - Using custom type declaration from server/types
import xss from 'xss-clean';
import hpp from 'hpp';
import mongoSanitize from 'express-mongo-sanitize';
import { v4 as uuidv4 } from 'uuid';
import express from 'express';

// IP Whitelisting Middleware
export const ipWhitelist = (allowedIPs: string[] = []) => {
  // Convert CIDR notation to IP range
  const isIPinRange = (ip: string, cidr: string): boolean => {
    try {
      if (!cidr.includes('/')) {
        return ip === cidr;
      }

      const [subnet, bits] = cidr.split('/');
      const ipToInt = (ip: string) => {
        return ip.split('.').reduce((ipInt, octet) => (ipInt << 8) + parseInt(octet, 10), 0) >>> 0;
      };

      const mask = ~(0xFFFFFFFF >>> parseInt(bits, 10));
      const ipInt = ipToInt(ip);
      const subnetInt = ipToInt(subnet);
      
      return (ipInt & mask) === (subnetInt & mask);
    } catch (err) {
      logger.error('IP range check failed', { error: err, ip, cidr });
      return false;
    }
  };

  return (req: Request, res: Response, next: NextFunction) => {
    // Skip if no IPs are whitelisted
    if (allowedIPs.length === 0) {
      return next();
    }

    const clientIP = req.ip || 
                    (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() || 
                    req.socket.remoteAddress;

    if (!clientIP) {
      logger.warn('Could not determine client IP', { headers: req.headers });
      return res.status(403).json({
        success: false,
        code: 'IP_VALIDATION_FAILED',
        message: 'Could not validate client IP'
      });
    }

    const isAllowed = allowedIPs.some(ip => isIPinRange(clientIP, ip));
    
    if (!isAllowed) {
      logger.warn('IP not in whitelist', { ip: clientIP, path: req.path });
      return res.status(403).json({
        success: false,
        code: 'IP_NOT_ALLOWED',
        message: 'Your IP address is not authorized to access this resource'
      });
    }

    next();
  };
};

// Request Signing Middleware
export const requestSigner = (options: {
  secret: string;
  signatureHeader?: string;
  timestampHeader?: string;
  windowMs?: number;
}) => {
  const {
    secret,
    signatureHeader = 'x-signature',
    timestampHeader = 'x-timestamp',
    windowMs = 5 * 60 * 1000 // 5 minutes
  } = options;

  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      const signature = req.headers[signatureHeader.toLowerCase()] as string;
      const timestamp = parseInt(req.headers[timestampHeader.toLowerCase()] as string, 10);

      // Check if required headers are present
      if (!signature || isNaN(timestamp)) {
        return res.status(400).json({
          success: false,
          code: 'SIGNATURE_REQUIRED',
          message: 'Request signature and timestamp are required'
        });
      }

      // Check timestamp (prevent replay attacks)
      const now = Date.now();
      if (Math.abs(now - timestamp) > windowMs) {
        return res.status(400).json({
          success: false,
          code: 'INVALID_TIMESTAMP',
          message: 'Request timestamp is too old or in the future'
        });
      }

      // Reconstruct the signed string
      const method = req.method.toUpperCase();
      const path = req.originalUrl || req.url;
      const body = Object.keys(req.body || {}).length > 0 
        ? JSON.stringify(req.body) 
        : '';
      
      const stringToSign = [
        method,
        path,
        timestamp,
        createHmac('sha256', secret)
          .update(body)
          .digest('hex')
      ].join('\n');

      // Generate expected signature
      const expectedSignature = createHmac('sha256', secret)
        .update(stringToSign)
        .digest('hex');

      // Compare signatures in constant time
      if (!timingSafeEqual(
        Buffer.from(signature, 'hex'),
        Buffer.from(expectedSignature, 'hex')
      )) {
        logger.warn('Invalid request signature', {
          path,
          method,
          receivedSignature: signature,
          expectedSignature
        });
        
        return res.status(401).json({
          success: false,
          code: 'INVALID_SIGNATURE',
          message: 'Invalid request signature'
        });
      }

      next();
    } catch (error) {
      logger.error('Request signing validation failed', { error });
      return res.status(500).json({
        success: false,
        code: 'SIGNATURE_VALIDATION_FAILED',
        message: 'Failed to validate request signature'
      });
    }
  };
};

// Use the existing AuthUser type from auth.ts

// API Key Authentication Middleware
export const apiKeyAuth = (requiredRoles: string[] = []) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      const apiKey = req.headers['x-api-key'] as string || req.query.api_key as string;
      
      if (!apiKey) {
        return res.status(401).json({
          success: false,
          code: 'API_KEY_REQUIRED',
          message: 'API key is required'
        });
      }

      const user = await ApiKeyService.getUserByApiKey(apiKey);
      
      if (!user) {
        return res.status(401).json({
          success: false,
          code: 'INVALID_API_KEY',
          message: 'Invalid or expired API key'
        });
      }

      // Check role if any is required
      if (requiredRoles.length > 0 && !requiredRoles.includes(user.role)) {
        return res.status(403).json({
          success: false,
          code: 'INSUFFICIENT_PERMISSIONS',
          message: 'Insufficient permissions to access this resource'
        });
      }

      // Attach user information to the request
      const authUser: AuthUser = {
        id: user.id.toString(),
        email: user.email || '',
        role: user.role || 'user',
        email_verified: false // Default to false since we don't have this information from the API key auth
      };
      req.user = authUser;
      
      next();
    } catch (error) {
      logger.error('API key validation failed', { error });
      return res.status(500).json({
        success: false,
        code: 'AUTHENTICATION_FAILED',
        message: 'Failed to authenticate request'
      });
    }
  };
};

// Security Headers Middleware
interface SecurityHeadersOptions {
  enableCSP?: boolean;
  enableHSTS?: boolean;
  enableXSS?: boolean;
  enableFrameOptions?: boolean;
  enableNoSniff?: boolean;
  cspDirectives?: Record<string, string[]>;
  maxRequestBodySize?: string;
}

export const securityHeaders = ({
  enableCSP = true,
  enableHSTS = true,
  enableXSS = true,
  enableFrameOptions = true,
  enableNoSniff = true,
  cspDirectives = {
    defaultSrc: ["'self'"],
    scriptSrc: ["'self'"],
    styleSrc: ["'self'"],
    imgSrc: ["'self'", 'data:'],
    connectSrc: ["'self'"],
    fontSrc: ["'self'"]
  },
  maxRequestBodySize = '10mb'
}: SecurityHeadersOptions = {}): RequestHandler[] => {
  return [
    // Parse JSON bodies with size limit
    express.json({
      limit: maxRequestBodySize,
      verify: (req: any, res: any, buf: Buffer) => {
        req.rawBody = buf.toString();
      },
    }),
    
    // Parse URL-encoded bodies with size limit
    express.urlencoded({
      extended: true,
      limit: maxRequestBodySize,
      parameterLimit: 100,
    }),

    // Security headers with individual middleware for better typing
    helmet(),
    enableCSP ? contentSecurityPolicy({
      useDefaults: true,
      directives: cspDirectives as any,
    }) : (req, res, next) => next(),
    crossOriginEmbedderPolicy(),
    crossOriginOpenerPolicy(),
    crossOriginResourcePolicy(),
    dnsPrefetchControl(),
    enableHSTS ? hsts({
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true,
    }) : (req, res, next) => next(),
    ieNoOpen(),
    enableNoSniff ? noSniff() : (req, res, next) => next(),
    originAgentCluster(),
    referrerPolicy({
      policy: 'strict-origin-when-cross-origin',
    }),
    enableXSS ? xssFilter() : (req, res, next) => next(),
    enableFrameOptions ? frameguard({ action: 'deny' }) : (req, res, next) => next(),
    hidePoweredBy(),

    // Prevent HTTP Parameter Pollution
    hpp({
      whitelist: [
        'filter',
        'sort',
        'limit',
        'page',
        'fields',
      ],
    }),

    // Sanitize request data
    xss(),

    // Prevent NoSQL injection
    mongoSanitize({
      onSanitize: ({ req, key }) => {
        logger.warn('NoSQL injection attempt detected', {
          ip: req.ip,
          method: req.method,
          url: req.originalUrl,
          key,
        });
      },
    }),

    // Add security response headers
    (req: Request, res: Response, next: NextFunction) => {
      // Add request ID
      req.id = req.get('X-Request-ID') || uuidv4();
      res.setHeader('X-Request-ID', req.id);
      
      // Security headers
      res.setHeader('X-Content-Type-Options', 'nosniff');
      res.setHeader('X-Frame-Options', 'DENY');
      res.setHeader('X-XSS-Protection', '1; mode=block');
      res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
      res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
      
      next();
    },

    // Request validation error handler
    ((err: any, req: Request, res: Response, next: NextFunction) => {
      if (err instanceof SyntaxError && 'body' in err) {
        return res.status(400).json({
          success: false,
          code: 'INVALID_JSON',
          message: 'Invalid JSON payload',
          requestId: req.id,
        });
      }
      next(err);
    }) as express.ErrorRequestHandler,

    // Rate limiting for auth endpoints
    rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // Limit each IP to 100 requests per windowMs
      standardHeaders: true,
      legacyHeaders: false,
      message: {
        success: false,
        code: 'TOO_MANY_REQUESTS',
        message: 'Too many requests, please try again later',
        requestId: (req: Request) => req.id,
      },
    }),
  ];
};

// ... (rest of the code remains the same)
