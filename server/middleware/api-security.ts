import type { Request, Response, NextFunction, RequestHandler } from 'express';
import rateLimit, { RateLimitRequestHandler } from 'express-rate-limit';
import { validationResult, ValidationChain, ValidationError } from 'express-validator';
import { defaultSecurityConfig } from '../config/security';
import { logger } from '../config/logger';

type RateLimitConfig = typeof defaultSecurityConfig.rateLimiting;
type RateLimitRule = RateLimitConfig['default'];

// Extended Request interface for internal use
interface ExtendedRequest extends Request {
  rateLimit?: {
    limit: number;
    current: number;
    remaining: number;
    resetTime?: Date;
  };
  apiKey?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any; // Allow additional properties
}

// Custom validation error type
// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface CustomValidationError {
  param?: string;
  msg: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value?: any;
  location?: string;
  nestedErrors?: ValidationError[];
}

// Rate limiting middleware
export const rateLimiter = (config = defaultSecurityConfig.rateLimiting) => {
  if (!config.enabled) {
    return (req: Request, res: Response, next: NextFunction) => next();
  }

  const defaultLimiter = rateLimit({
    windowMs: config.default.windowMs,
    max: config.default.maxRequests,
    skipFailedRequests: config.default.skipFailedRequests,
    skipSuccessfulRequests: config.default.skipSuccessfulRequests,
    keyGenerator: (req) => {
      // Ensure we always return a string
      const key = config.default.keyGenerator?.(req) || req.ip;
      return key || 'unknown';
    },
    handler: (req, res) => {
      if (config.default.handler) {
        return config.default.handler(req, res);
      }
      res.status(429).json({
        success: false,
        message: config.default.message || 'Too many requests, please try again later.',
        code: 'RATE_LIMIT_EXCEEDED',
        retryAfter: (() => {
          const rateLimitInfo = (req as ExtendedRequest).rateLimit;
          return rateLimitInfo?.resetTime
            ? Math.ceil((rateLimitInfo.resetTime.getTime() - Date.now()) / 1000)
            : undefined;
        })()
      });
    },
    standardHeaders: true,
    legacyHeaders: false,
    message: 'Too many requests, please try again later.',
    statusCode: 429
  });

  // Create rate limiters for specific routes
  const limiters = new Map<string, RateLimitRequestHandler>();
  
  // Convert rules to array and process them
  const ruleEntries = Object.entries(config.rules) as [string, RateLimitRule][];
  
  for (const [key, rule] of ruleEntries) {
    const limiter = rateLimit({
      windowMs: rule.windowMs,
      max: rule.maxRequests,
      skipFailedRequests: rule.skipFailedRequests,
      skipSuccessfulRequests: rule.skipSuccessfulRequests,
      keyGenerator: (req) => {
        const key = rule.keyGenerator?.(req) || req.ip;
        return key || 'unknown';
      },
      handler: (req, res) => {
        if (rule.handler) {
          return rule.handler(req, res);
        }
        res.status(429).json({
          success: false,
          message: rule.message || 'Too many requests, please try again later.',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: (() => {
            const rateLimitInfo = (req as ExtendedRequest).rateLimit;
            return rateLimitInfo?.resetTime
              ? Math.ceil((rateLimitInfo.resetTime.getTime() - Date.now()) / 1000)
              : undefined;
          })()
        });
      },
      standardHeaders: true,
      legacyHeaders: false,
      message: rule.message || 'Too many requests, please try again later.',
      statusCode: 429
    });
    
    limiters.set(key, limiter);
  }

  // Return appropriate rate limiter based on route
  return (req: Request, res: Response, next: NextFunction) => {
    // Check if there's a specific rate limiter for this route
    const routeKey = Array.from(limiters.keys()).find(key => req.path.includes(key));
    if (routeKey) {
      const limiter = limiters.get(routeKey);
      if (limiter) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return limiter(req as any, res as any, next);
      }
    }
    // Use default rate limiter
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return defaultLimiter(req as any, res as any, next);
  };
};

// Request validation middleware
export const validateRequest = (validations: ValidationChain[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Run all validations
      await Promise.all(validations.map(validation => validation.run(req)));
      
      const errors = validationResult(req);
      if (errors.isEmpty()) {
        return next();
      }

      // Format validation errors
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const errorMessages = errors.array().map((err: any) => ({
        field: err.param || 'unknown',
        message: err.msg || 'Validation error',
        value: err.value,
        location: err.location || 'body',
        ...(err.nestedErrors && { nestedErrors: err.nestedErrors })
      }));

      // Log validation errors
      logger.warn('Request validation failed', {
        path: req.path,
        method: req.method,
        errors: errorMessages,
        ip: req.ip,
        userAgent: req.get('user-agent')
      });

      return res.status(400).json({
        success: false,
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        errors: errorMessages
      });
    } catch (error) {
      logger.error('Validation middleware error', { error });
      next(error);
    }
  };
};

// Response formatting middleware
export const formatResponse: RequestHandler = (req, res, next) => {
  // Store original json method
  const originalJson = res.json;

  // Override res.json
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  res.json = function (data?: any): Response {
    // Skip formatting for error responses or if already formatted
    if (res.statusCode >= 400 || (data && typeof data === 'object' && 'success' in data)) {
      return originalJson.call(this, data);
    }

    // Handle different response types
    let responseData = data;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const meta: Record<string, any> = {
      timestamp: new Date().toISOString(),
      version: defaultSecurityConfig.api.version,
      path: req.path,
      method: req.method
    };

    // Handle pagination
    if (data && typeof data === 'object' && 'data' in data && 'pagination' in data) {
      responseData = data.data;
      meta.pagination = data.pagination;
    }

    // Format successful response
    const response = {
      success: true,
      data: responseData,
      meta
    };

    // Set standard headers
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('X-Frame-Options', 'DENY');
    
    // Add rate limit headers if available
    const rateLimitInfo = (req as ExtendedRequest).rateLimit;
    if (rateLimitInfo) {
      res.setHeader('X-RateLimit-Limit', rateLimitInfo.limit.toString());
      res.setHeader('X-RateLimit-Remaining', rateLimitInfo.remaining.toString());
      if (rateLimitInfo.resetTime) {
        res.setHeader('X-RateLimit-Reset', Math.ceil((rateLimitInfo.resetTime.getTime() / 1000)).toString());
      }
    }

    return originalJson.call(this, response);
  };

  next();
};

// API key authentication
export const apiKeyAuth: RequestHandler = async (req, res, next) => {
  try {
    // Get API key from header, query param, or cookie
    const apiKey = (
      req.headers['x-api-key'] ||
      req.query.api_key ||
      req.cookies?.api_key
    ) as string | undefined;
    
    if (!apiKey) {
      logger.warn('API key missing', {
        path: req.path,
        method: req.method,
        ip: req.ip
      });
      
      return res.status(401).json({
        success: false,
        message: 'API key is required',
        code: 'API_KEY_REQUIRED',
        docs: 'https://docs.your-api.com/authentication'
      });
    }

    // In a real application, validate the API key against your database
    // This is a simplified example
    const isValid = await validateApiKey(apiKey);
    
    if (!isValid) {
      logger.warn('Invalid API key', {
        path: req.path,
        method: req.method,
        ip: req.ip,
        key: maskApiKey(apiKey)
      });
      
      return res.status(403).json({
        success: false,
        message: 'Invalid API key',
        code: 'INVALID_API_KEY',
        docs: 'https://docs.your-api.com/authentication'
      });
    }

    // Add API key info to request for later use
    (req as ExtendedRequest).apiKey = apiKey;
    next();
  } catch (error) {
    logger.error('API key validation error', { error });
    next(error);
  }
};

// Helper function to validate API key (replace with actual implementation)
async function validateApiKey(apiKey: string): Promise<boolean> {
  // In a real application, you would check this against your database
  // This is a simplified example
  return apiKey === process.env.API_KEY;
}

// Helper function to mask API key for logging
function maskApiKey(key: string, visibleChars = 4): string {
  if (!key) return '';
  if (key.length <= visibleChars * 2) return '****';
  
  const first = key.substring(0, visibleChars);
  const last = key.substring(key.length - visibleChars);
  return `${first}...${last}`;
}

// Request logging middleware
export const requestLogger: RequestHandler = (req, res, next) => {
  const start = Date.now();
  const { method, originalUrl, ip, headers, body } = req;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { query, params } = req;
  
  // Skip logging for health checks
  if (originalUrl === '/health' || originalUrl === '/status') {
    return next();
  }
  
  // Log request details
  const requestId = (req.headers['x-request-id'] as string) || Date.now().toString(36);
  
  // Add request ID to response
  res.setHeader('X-Request-ID', requestId);
  
  // Log request start
  logger.info('API Request Start', {
    requestId,
    method,
    url: originalUrl,
    ip,
    userAgent: headers['user-agent'] || '',
    timestamp: new Date().toISOString()
  });
  
  // Log request body (with sensitive data redacted)
  if (Object.keys(body).length > 0) {
    logger.debug('Request Body', {
      requestId,
      body: redactSensitiveData(body)
    });
  }
  
  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - start;
    const { statusCode } = res;
    const contentLength = res.get('content-length') || '0';
    
    const logData = {
      requestId,
      method,
      url: originalUrl,
      ip,
      status: statusCode,
      duration: `${duration}ms`,
      contentLength,
      userAgent: headers['user-agent'] || '',
      timestamp: new Date().toISOString()
    };

    // Log based on status code
    if (statusCode >= 500) {
      logger.error('API Request Error', logData);
    } else if (statusCode >= 400) {
      logger.warn('API Client Error', logData);
    } else {
      logger.info('API Request Complete', logData);
    }
  });

  next();
};

// Helper function to redact sensitive data from logs
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function redactSensitiveData(obj: Record<string, any>): Record<string, any> {
  const sensitiveFields = [
    'password',
    'newPassword',
    'currentPassword',
    'confirmPassword',
    'token',
    'accessToken',
    'refreshToken',
    'apiKey',
    'authorization'
  ];
  
  if (!obj || typeof obj !== 'object') return obj;
  
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result: Record<string, any> = Array.isArray(obj) ? [] : {};
  
  for (const [key, value] of Object.entries(obj)) {
    if (sensitiveFields.includes(key.toLowerCase())) {
      result[key] = '***REDACTED***';
    } else if (value && typeof value === 'object') {
      result[key] = redactSensitiveData(value);
    } else {
      result[key] = value;
    }
  }
  
  return result;
}

export default {
  rateLimiter,
  validateRequest,
  formatResponse,
  apiKeyAuth,
  requestLogger
};
