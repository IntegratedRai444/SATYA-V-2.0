import { Request, Response } from 'express';
import { logger } from './logger';

export interface RateLimitRule {
  windowMs: number;
  maxRequests: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  message?: string;
  keyGenerator?: (req: Request) => string;
  handler?: (req: Request, res: Response) => void;
  onLimitReached?: (req: Request, res: Response) => void;
}

export interface SecurityConfig {
  // JWT Configuration
  jwt: {
    secret: string;
    accessTokenExpiry: string | number;
    refreshTokenExpiry: string | number;
    issuer: string;
    audience: string;
  };
  
  // Rate Limiting
  rateLimiting: {
    enabled: boolean;
    default: RateLimitRule;
    rules: Record<string, RateLimitRule>;
  };
  
  // CORS
  cors: {
    enabled: boolean;
    allowedOrigins: string[];
    allowedMethods: string[];
    allowedHeaders: string[];
    exposedHeaders: string[];
    credentials: boolean;
    maxAge: number;
  };
  
  // Security Headers
  securityHeaders: {
    enabled: boolean;
    hsts: {
      maxAge: number;
      includeSubDomains: boolean;
      preload: boolean;
    };
    contentSecurityPolicy: {
      directives: Record<string, string[]>;
      reportUri?: string;
    };
    featurePolicy: {
      features: Record<string, string[]>;
    };
  };
  
  // CSRF Protection
  csrf: {
    enabled: boolean;
    cookie: {
      name: string;
      httpOnly: boolean;
      secure: boolean;
      sameSite: 'strict' | 'lax' | 'none';
      maxAge: number;
      path?: string;
      domain?: string;
    };
    protectedMethods: string[];
    exemptRoutes: string[];
    headerName: string;
    fieldName: string;
    maxAge: number;
  };
  
  // Password Policies
  passwordPolicy: {
    minLength: number;
    requireUppercase: boolean;
    requireLowercase: boolean;
    requireNumbers: boolean;
    requireSpecialChars: boolean;
    maxAge: number; // days
    history: number; // remember last N passwords
  };
  
  // Session Management
  session: {
    name: string;
    secret: string;
    resave: boolean;
    saveUninitialized: boolean;
    cookie: {
      httpOnly: boolean;
      secure: boolean;
      sameSite: 'strict' | 'lax' | 'none';
      maxAge: number;
    };
    rolling: boolean;
  };
  
  // API Security
  api: {
    version: string;
    prefix: string;
    jsonLimit: string;
    urlEncoded: boolean;
    requestSizeLimit: string;
  };
  
  // Logging
  logging: {
    securityEvents: boolean;
    failedLoginAttempts: boolean;
    sensitiveOperations: boolean;
  };
  
  // Security Headers
  headers: {
    // XSS Protection
    xssProtection: boolean;
    // MIME type sniffing protection
    noSniff: boolean;
    // Clickjacking protection
    xFrameOptions: boolean;
    // MIME type security
    xContentTypeOptions: boolean;
    // Referrer policy
    referrerPolicy: string;
    // Permissions policy (previously Feature Policy)
    permissionsPolicy: Record<string, string[]>;
    // Content Security Policy (CSP)
    csp: {
      defaultSrc: string[];
      scriptSrc: string[];
      styleSrc: string[];
      imgSrc: string[];
      connectSrc: string[];
      fontSrc: string[];
      objectSrc: string[];
      mediaSrc: string[];
      frameSrc: string[];
      frameAncestors: string[];
      formAction: string[];
      baseUri: string[];
      upgradeInsecureRequests: boolean;
      blockAllMixedContent: boolean;
      reportUri?: string;
      reportOnly: boolean;
    };
    // HTTP Strict Transport Security (HSTS)
    hsts: {
      maxAge: number;
      includeSubDomains: boolean;
      preload: boolean;
    };
    // Expect-CT (Certificate Transparency)
    expectCt: {
      maxAge: number;
      enforce: boolean;
      reportUri?: string;
    };
  };
}

// Default security configuration
export const defaultSecurityConfig: SecurityConfig = {
  // JWT Configuration
  jwt: {
    secret: process.env.JWT_SECRET || 'your-secret-key',
    accessTokenExpiry: '15m', // 15 minutes
    refreshTokenExpiry: '7d', // 7 days
    issuer: 'satya-ai',
    audience: 'satya-ai-client'
  },
  
  // Rate Limiting
  rateLimiting: {
    enabled: true,
    default: {
      windowMs: 15 * 60 * 1000, // 15 minutes
      maxRequests: 100,
      skipSuccessfulRequests: true,
      skipFailedRequests: false,
      message: 'Too many requests, please try again later.'
    },
    rules: {
      // Authentication endpoints (login, register, password reset)
      auth: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        maxRequests: 5,
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
        message: 'Too many login attempts. Please try again later.',
        keyGenerator: (req) => `auth_${req.ip}`,
        handler: (req, res) => {
          logger.warn(`Rate limit reached for authentication endpoint from IP: ${req.ip}`);
          res.status(429).json({
            success: false,
            message: 'Too many login attempts. Please try again in 15 minutes.',
            retryAfter: 15 * 60 // 15 minutes in seconds
          });
        }
      },
      // Public API endpoints
      publicApi: {
        windowMs: 60 * 1000, // 1 minute
        maxRequests: 60,
        skipSuccessfulRequests: true,
        message: 'Too many requests, please try again later.'
      },
      // File upload endpoints
      fileUpload: {
        windowMs: 5 * 60 * 1000, // 5 minutes
        maxRequests: 10,
        message: 'Too many file uploads. Please try again later.'
      },
      // Sensitive operations (password changes, email updates)
      sensitiveOperations: {
        windowMs: 60 * 60 * 1000, // 1 hour
        maxRequests: 5,
        message: 'Too many attempts. Please try again later.'
      },
      // API key based rate limiting
      apiKey: {
        windowMs: 60 * 1000, // 1 minute
        maxRequests: 100,
        keyGenerator: (req) => req.headers['x-api-key'] as string || 'anonymous',
        message: 'API rate limit exceeded. Please check your API key usage.'
      }
    }
  },
  
  // CORS
  cors: {
    enabled: true,
    allowedOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000'],
    allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
    exposedHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
    credentials: true,
    maxAge: 600 // 10 minutes
  },
  
  // Security Headers
  headers: {
    // Basic headers
    xssProtection: true,
    noSniff: true,
    xFrameOptions: true,
    xContentTypeOptions: true,
    referrerPolicy: 'strict-origin-when-cross-origin',
    
    // Permissions Policy (previously Feature Policy)
    permissionsPolicy: {
      'geolocation': ["'none'"],
      'camera': ["'none'"],
      'microphone': ["'none'"],
      'payment': ["'none'"],
      'fullscreen': ["'self'"],
      'autoplay': ["'self'"],
      'sync-xhr': ["'self'"],
      'accelerometer': ["'none'"],
      'gyroscope': ["'none'"],
      'magnetometer': ["'none'"],
      'picture-in-picture': ["'none'"],
      'usb': ["'none'"],
      'vr': ["'none'"],
      'wake-lock': ["'none'"],
      'screen-wake-lock': ["'none'"],
      'web-share': ["'none'"],
      'display-capture': ["'none'"],
      'clipboard-read': ["'none'"],
      'clipboard-write': ["'none'"],
      'gamepad': ["'none'"],
      'speaker-selection': ["'none'"],
      'conversion-measurement': ["'none'"],
      'focus-without-user-activation': ["'none'"],
      'hid': ["'none'"],
      'idle-detection': ["'none'"],
      'serial': ["'none'"],
      'sync-script': ["'none'"],
      'trust-token-redemption': ["'none'"],
      'vertical-scroll': ["'none'"]
    },
    
    // Content Security Policy (CSP)
    csp: {
      defaultSrc: ["'self'"],
      scriptSrc: [
        "'self'",
        "'unsafe-inline'",
        "'unsafe-eval'"
      ],
      styleSrc: [
        "'self'",
        "'unsafe-inline'"
      ],
      imgSrc: [
        "'self'",
        'data:',
        'https:',
        'http:'
      ],
      connectSrc: [
        "'self'",
        'https://api.example.com',
        'wss://ws.example.com'
      ],
      fontSrc: [
        "'self'",
        'data:',
        'https:'
      ],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
      frameAncestors: ["'none'"],
      formAction: ["'self'"],
      baseUri: ["'self'"],
      upgradeInsecureRequests: true,
      blockAllMixedContent: true,
      reportOnly: false
    },
    
    // HTTP Strict Transport Security (HSTS)
    hsts: {
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true
    },
    
    // Expect-CT (Certificate Transparency)
    expectCt: {
      maxAge: 86400, // 1 day
      enforce: true,
      reportUri: '/report-ct-violation'
    }
  },
  
  // CSRF Protection
  csrf: {
    enabled: true,
    cookie: {
      name: 'XSRF-TOKEN',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 86400000, // 24 hours
      domain: process.env.COOKIE_DOMAIN || undefined,
      path: '/',
    },
    protectedMethods: ['POST', 'PUT', 'PATCH', 'DELETE'],
    exemptRoutes: [
      '/api/webhooks/',
      '/api/auth/csrf-token',
      '/health'
    ],
    headerName: 'X-CSRF-TOKEN',
    fieldName: '_csrf',
    maxAge: 24 * 60 * 60 * 1000
  },
  
  // Password Policies
  passwordPolicy: {
    minLength: 12,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: true,
    maxAge: 90, // days
    history: 5 // remember last 5 passwords
  },
  
  // Session Management
  session: {
    name: 'satya.sid',
    secret: process.env.SESSION_SECRET || 'your-session-secret',
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 24 * 60 * 60 * 1000 // 24 hours
    },
    rolling: true
  },
  
  // API Security
  api: {
    version: 'v1',
    prefix: '/api',
    jsonLimit: '10mb',
    urlEncoded: true,
    requestSizeLimit: '10mb'
  },
  
  // Logging
  logging: {
    securityEvents: true,
    failedLoginAttempts: true,
    sensitiveOperations: true
  },
  
  // Security Headers
  headers: {
    xssProtection: true,
    noSniff: true,
    xFrameOptions: true,
    xContentTypeOptions: true,
    referrerPolicy: 'strict-origin-when-cross-origin',
    permissionsPolicy: {
      'geolocation': ["'none'"],
      'camera': ["'none'"],
      'microphone': ["'none'"],
      'payment': ["'none'"]
    }
  }
};

// Export the configuration
export const getSecurityConfig = () => defaultSecurityConfig;

export default defaultSecurityConfig;
