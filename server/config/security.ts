import type { Request, Response } from 'express';

// Rate limiting configuration
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
    algorithm?: string;
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
      directives: Record<string, (string | boolean)[]>;
      reportOnly?: boolean;
      reportUri?: string;
    };
    featurePolicy?: {
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
    maxAge: number;
    history: number;
  };
  
  // Session Management
  session: {
    name?: string;
    secret: string;
    resave: boolean;
    saveUninitialized: boolean;
    cookie: {
      httpOnly: boolean;
      secure: boolean;
      sameSite: 'strict' | 'lax' | 'none';
      maxAge?: number;
      path?: string;
      domain?: string;
    };
    rolling?: boolean;
    unset?: 'destroy' | 'keep';
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
    xssProtection: boolean;
    noSniff: boolean;
    xFrameOptions: boolean;
    xContentTypeOptions: boolean;
    referrerPolicy: string;
    permissionsPolicy: Record<string, string[]>;
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
    hsts: {
      maxAge: number;
      includeSubDomains: boolean;
      preload: boolean;
    };
    expectCt: {
      maxAge: number;
      enforce: boolean;
      reportUri?: string;
    };
  };
}

// Validate required environment variables
const requiredEnvVars = [
  'JWT_SECRET',
  'JWT_ISSUER',
  'JWT_AUDIENCE',
  'SESSION_SECRET',
  'CSRF_TOKEN_SECRET'
];

// Check for missing environment variables
const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingVars.length > 0 && process.env.NODE_ENV === 'production') {
  throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
}

// CRITICAL: JWT Secret validation - no fallback defaults for production security
const jwtSecret = process.env.JWT_SECRET;
if (!jwtSecret) {
  throw new Error('JWT_SECRET environment variable is required for security');
}

// Default security configuration
export const defaultSecurityConfig: SecurityConfig = {
  // JWT Configuration
  jwt: {
    secret: jwtSecret,
    accessTokenExpiry: process.env.JWT_ACCESS_EXPIRY || '15m', // 15 minutes
    refreshTokenExpiry: process.env.JWT_REFRESH_EXPIRY || '7d', // 7 days
    issuer: process.env.JWT_ISSUER || 'satya-ai',
    audience: process.env.JWT_AUDIENCE || 'satya-ai-client',
    algorithm: 'HS256'
  },

  // Rate Limiting
  rateLimiting: {
    enabled: process.env.RATE_LIMIT_ENABLED !== 'false',
    default: {
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10), // 15 minutes
      maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10),
      message: 'Too many requests, please try again later.',
      keyGenerator: (req: Request) => {
        // Rate limit by IP and user ID if authenticated
        const ip = req.ip || 'unknown-ip';
        const userId = req.user?.id || 'anonymous';
        return `${ip}:${userId}`;
      }
    },
    rules: {
      // Authentication endpoints
      login: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        maxRequests: 5,
        message: 'Too many login attempts. Please try again later.'
      },
      // Password reset endpoints
      passwordReset: {
        windowMs: 60 * 60 * 1000, // 1 hour
        maxRequests: 3,
        message: 'Too many password reset attempts. Please try again later.'
      },
      // File upload endpoints
      fileUpload: {
        windowMs: 5 * 60 * 1000, // 5 minutes
        maxRequests: 10,
        message: 'Too many file uploads. Please try again later.'
      },
      // API endpoints
      api: {
        windowMs: 60 * 1000, // 1 minute
        maxRequests: 100,
        message: 'API rate limit exceeded. Please try again later.'
      }
    }
  },

  // CORS Configuration
  cors: {
    enabled: true,
    allowedOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:5173'],
    allowedMethods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'X-API-Version'],
    exposedHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
    credentials: true,
    maxAge: 600 // 10 minutes
  },

  // Security Headers
  securityHeaders: {
    enabled: process.env.SECURITY_HEADERS_ENABLED !== 'false',
    // HSTS - Only enable in production with HTTPS
    hsts: {
      maxAge: 31536000, // 1 year in seconds
      includeSubDomains: true,
      preload: process.env.NODE_ENV === 'production'
    },
    // Content Security Policy
    contentSecurityPolicy: {
      directives: {
        'default-src': ["'self'"],
        'script-src': [
          "'self'",
          // Add hashes or nonces for inline scripts in production
          ...(process.env.NODE_ENV === 'development' ? ["'unsafe-inline'", "'unsafe-eval'"] : [])
        ],
        'style-src': ["'self'", "'unsafe-inline'"],
        'img-src': ["'self'", "data:", "https://*"],
        'connect-src': ["'self'", "https://*"],
        'font-src': ["'self'", "data:"],
        'object-src': ["'none'"],
        'media-src': ["'self'"],
        'frame-src': ["'self'"],
        'frame-ancestors': ["'self'"],
        'form-action': ["'self'"],
        'base-uri': ["'self'"]
      },
      reportOnly: process.env.NODE_ENV === 'development',
      reportUri: process.env.CSP_REPORT_URI
    },
    featurePolicy: {
      features: {
        camera: ["'none'"],
        microphone: ["'none'"],
        geolocation: ["'none'"],
        fullscreen: ["'self'"],
        payment: ["'none'"]
      }
    }
  },

  // CSRF Protection
  csrf: {
    enabled: process.env.NODE_ENV !== 'test',
    cookie: {
      name: 'XSRF-TOKEN',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
      path: '/',
      domain: process.env.COOKIE_DOMAIN
    },
    protectedMethods: ['POST', 'PUT', 'PATCH', 'DELETE'],
    exemptRoutes: [
      '/api/webhooks',
      '/api/auth/csrf-token',
      '/health'
    ],
    headerName: 'X-CSRF-TOKEN',
    fieldName: '_csrf',
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  },

  // Password Policy
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
    name: 'sessionId',
    secret: process.env.SESSION_SECRET || 'your-session-secret',
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 * 1000 // 1 week
    },
    rolling: true
  },

  // API Configuration
  api: {
    version: 'v1',
    prefix: '/api',
    jsonLimit: '10mb',
    urlEncoded: true,
    requestSizeLimit: '10mb'
  },

  // Logging Configuration
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
    
    csp: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'"],
      fontSrc: ["'self'", 'data:'],
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
    
    hsts: {
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true
    },
    
    expectCt: {
      maxAge: 86400, // 1 day
      enforce: true,
      reportUri: '/report-ct-violation'
    }
  }
};

// Helper function to get security configuration
export function getSecurityConfig(): SecurityConfig {
  return defaultSecurityConfig;
}

export default defaultSecurityConfig;
