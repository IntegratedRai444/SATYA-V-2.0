import type { Request, Response, NextFunction } from 'express';
import { createHmac, timingSafeEqual } from 'crypto';
import { logger } from '../config/logger';
import { apiKeyService } from '../services/apiKeyService';

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

// API Key Authentication Middleware
export const apiKeyAuth = (requiredPermissions: string[] = []) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      const apiKey = req.headers['x-api-key'] as string;
      
      if (!apiKey) {
        return res.status(401).json({
          success: false,
          code: 'API_KEY_REQUIRED',
          message: 'API key is required'
        });
      }

      const { isValid, keyData } = await apiKeyService.verifyKey(apiKey);
      
      if (!isValid || !keyData) {
        return res.status(401).json({
          success: false,
          code: 'INVALID_API_KEY',
          message: 'Invalid or expired API key'
        });
      }

      // Check permissions if any are required
      if (requiredPermissions.length > 0) {
        const hasPermission = requiredPermissions.every(permission => 
          keyData.permissions.includes(permission)
        );

        if (!hasPermission) {
          return res.status(403).json({
            success: false,
            code: 'INSUFFICIENT_PERMISSIONS',
            message: 'Insufficient permissions to access this resource'
          });
        }
      }

      // Attach key information to the request
      req.apiKey = apiKey;
      (req as any).apiKeyData = keyData;
      
      // Track API key usage
      apiKeyService.trackApiKeyUsage(apiKey, req);
      
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
export const securityHeaders = (options: {
  enableCSP?: boolean;
  enableHSTS?: boolean;
  enableXSS?: boolean;
  enableFrameOptions?: boolean;
  enableNoSniff?: boolean;
  cspDirectives?: Record<string, string[]>;
}) => {
  const {
    enableCSP = true,
    enableHSTS = true,
    enableXSS = true,
    enableFrameOptions = true,
    enableNoSniff = true,
    cspDirectives = {
      'default-src': ["'self'"],
      'script-src': ["'self'"],
      'style-src': ["'self'"],
      'img-src': ["'self'"],
      'connect-src': ["'self'"],
      'font-src': ["'self'"],
      'object-src': ["'none'"],
      'media-src': ["'self'"],
      'frame-src': ["'none'"],
      'worker-src': ["'self'"],
      'child-src': ["'self'"]
    }
  } = options;

  return (req: Request, res: Response, next: NextFunction) => {
    // Content Security Policy
    if (enableCSP) {
      const csp = Object.entries(cspDirectives)
        .map(([directive, sources]) => {
          return `${directive} ${sources.join(' ')}`;
        })
        .join('; ');
      
      res.setHeader('Content-Security-Policy', csp);
      res.setHeader('X-Content-Security-Policy', csp); // For older browsers
    }

    // HTTP Strict Transport Security
    if (enableHSTS) {
      res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
    }

    // XSS Protection
    if (enableXSS) {
      res.setHeader('X-XSS-Protection', '1; mode=block');
    }

    // Frame Options
    if (enableFrameOptions) {
      res.setHeader('X-Frame-Options', 'DENY');
    }

    // No Sniff
    if (enableNoSniff) {
      res.setHeader('X-Content-Type-Options', 'nosniff');
    }

    // Additional security headers
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
    
    next();
  };
};
