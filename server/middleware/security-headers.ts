import { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';

interface SecurityConfig {
  contentSecurityPolicy?: {
    enabled: boolean;
    directives?: Record<string, string[]>;
    reportUri?: string;
  };
  hsts?: {
    enabled: boolean;
    maxAge?: number;
    includeSubDomains?: boolean;
    preload?: boolean;
  };
  frameOptions?: 'DENY' | 'SAMEORIGIN' | 'ALLOW-FROM';
  contentTypeOptions?: boolean;
  referrerPolicy?: string;
  permissionsPolicy?: Record<string, string[]>;
  crossOriginEmbedderPolicy?: 'require-corp' | 'credentialless';
  crossOriginOpenerPolicy?: 'same-origin' | 'same-origin-allow-popups' | 'unsafe-none';
  crossOriginResourcePolicy?: 'same-site' | 'same-origin' | 'cross-origin';
}

const defaultSecurityConfig: SecurityConfig = {
  contentSecurityPolicy: {
    enabled: true,
    directives: {
      'default-src': ["'self'"],
      'script-src': ["'self'", "'unsafe-inline'"],
      'style-src': ["'self'", "'unsafe-inline'"],
      'img-src': ["'self'", 'data:', 'https:'],
      'font-src': ["'self'", 'https:'],
      'connect-src': ["'self'", 'ws:', 'wss:'],
      'media-src': ["'self'"],
      'object-src': ["'none'"],
      'frame-src': ["'none'"],
      'base-uri': ["'self'"],
      'form-action': ["'self'"],
      'frame-ancestors': ["'none'"],
      'upgrade-insecure-requests': []
    }
  },
  hsts: {
    enabled: true,
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true
  },
  frameOptions: 'DENY',
  contentTypeOptions: true,
  referrerPolicy: 'strict-origin-when-cross-origin',
  permissionsPolicy: {
    'camera': ["'self'"],
    'microphone': ["'self'"],
    'geolocation': ["'none'"],
    'payment': ["'none'"],
    'usb': ["'none'"],
    'magnetometer': ["'none'"],
    'gyroscope': ["'none'"],
    'accelerometer': ["'none'"]
  },
  crossOriginEmbedderPolicy: 'require-corp',
  crossOriginOpenerPolicy: 'same-origin',
  crossOriginResourcePolicy: 'same-origin'
};

class SecurityHeaders {
  private config: SecurityConfig;
  private nonces: Map<string, string> = new Map();

  constructor(config: Partial<SecurityConfig> = {}) {
    this.config = { ...defaultSecurityConfig, ...config };
  }

  // Generate nonce for CSP
  private generateNonce(): string {
    return crypto.randomBytes(16).toString('base64');
  }

  // Build CSP header value
  private buildCSPHeader(nonce?: string): string {
    if (!this.config.contentSecurityPolicy?.enabled || !this.config.contentSecurityPolicy.directives) {
      return '';
    }

    const directives = this.config.contentSecurityPolicy.directives;
    const cspParts: string[] = [];

    for (const [directive, values] of Object.entries(directives)) {
      if (values.length === 0) {
        cspParts.push(directive);
      } else {
        let directiveValues = values.join(' ');
        
        // Add nonce to script-src and style-src if provided
        if (nonce && (directive === 'script-src' || directive === 'style-src')) {
          directiveValues += ` 'nonce-${nonce}'`;
        }
        
        cspParts.push(`${directive} ${directiveValues}`);
      }
    }

    // Add report-uri if configured
    if (this.config.contentSecurityPolicy.reportUri) {
      cspParts.push(`report-uri ${this.config.contentSecurityPolicy.reportUri}`);
    }

    return cspParts.join('; ');
  }

  // Build Permissions Policy header
  private buildPermissionsPolicyHeader(): string {
    if (!this.config.permissionsPolicy) return '';

    const policies: string[] = [];
    for (const [feature, allowlist] of Object.entries(this.config.permissionsPolicy)) {
      if (allowlist.length === 0) {
        policies.push(`${feature}=()`);
      } else {
        policies.push(`${feature}=(${allowlist.join(' ')})`);
      }
    }

    return policies.join(', ');
  }

  // Main middleware function
  middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      // Generate nonce for this request
      const nonce = this.generateNonce();
      this.nonces.set(req.ip || 'unknown', nonce);

      // Store nonce in response locals for use in templates
      res.locals.nonce = nonce;

      // Content Security Policy
      if (this.config.contentSecurityPolicy?.enabled) {
        const cspHeader = this.buildCSPHeader(nonce);
        if (cspHeader) {
          res.setHeader('Content-Security-Policy', cspHeader);
        }
      }

      // HTTP Strict Transport Security
      if (this.config.hsts?.enabled && req.secure) {
        let hstsValue = `max-age=${this.config.hsts.maxAge || 31536000}`;
        if (this.config.hsts.includeSubDomains) {
          hstsValue += '; includeSubDomains';
        }
        if (this.config.hsts.preload) {
          hstsValue += '; preload';
        }
        res.setHeader('Strict-Transport-Security', hstsValue);
      }

      // X-Frame-Options
      if (this.config.frameOptions) {
        res.setHeader('X-Frame-Options', this.config.frameOptions);
      }

      // X-Content-Type-Options
      if (this.config.contentTypeOptions) {
        res.setHeader('X-Content-Type-Options', 'nosniff');
      }

      // Referrer Policy
      if (this.config.referrerPolicy) {
        res.setHeader('Referrer-Policy', this.config.referrerPolicy);
      }

      // Permissions Policy
      const permissionsPolicy = this.buildPermissionsPolicyHeader();
      if (permissionsPolicy) {
        res.setHeader('Permissions-Policy', permissionsPolicy);
      }

      // Cross-Origin Embedder Policy
      if (this.config.crossOriginEmbedderPolicy) {
        res.setHeader('Cross-Origin-Embedder-Policy', this.config.crossOriginEmbedderPolicy);
      }

      // Cross-Origin Opener Policy
      if (this.config.crossOriginOpenerPolicy) {
        res.setHeader('Cross-Origin-Opener-Policy', this.config.crossOriginOpenerPolicy);
      }

      // Cross-Origin Resource Policy
      if (this.config.crossOriginResourcePolicy) {
        res.setHeader('Cross-Origin-Resource-Policy', this.config.crossOriginResourcePolicy);
      }

      // Additional security headers
      res.setHeader('X-DNS-Prefetch-Control', 'off');
      res.setHeader('X-Download-Options', 'noopen');
      res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');
      res.setHeader('X-XSS-Protection', '1; mode=block');

      // Remove server information
      res.removeHeader('X-Powered-By');
      res.removeHeader('Server');

      next();
    };
  }

  // Get nonce for a request
  getNonce(ip: string): string | undefined {
    return this.nonces.get(ip);
  }

  // Update configuration
  updateConfig(newConfig: Partial<SecurityConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  // Get current configuration
  getConfig(): SecurityConfig {
    return { ...this.config };
  }
}

// CSRF Protection
export class CSRFProtection {
  private tokens: Map<string, { token: string; expires: number }> = new Map();
  private tokenExpiry: number = 60 * 60 * 1000; // 1 hour

  constructor(tokenExpiry?: number) {
    if (tokenExpiry) {
      this.tokenExpiry = tokenExpiry;
    }

    // Clean up expired tokens every 10 minutes
    setInterval(() => {
      this.cleanup();
    }, 10 * 60 * 1000);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, value] of Array.from(this.tokens.entries())) {
      if (now > value.expires) {
        this.tokens.delete(key);
      }
    }
  }

  private generateToken(): string {
    return crypto.randomBytes(32).toString('hex');
  }

  // Generate CSRF token for session
  generateTokenForSession(sessionId: string): string {
    const token = this.generateToken();
    this.tokens.set(sessionId, {
      token,
      expires: Date.now() + this.tokenExpiry
    });
    return token;
  }

  // Validate CSRF token
  validateToken(sessionId: string, providedToken: string): boolean {
    const storedToken = this.tokens.get(sessionId);
    
    if (!storedToken) {
      return false;
    }

    if (Date.now() > storedToken.expires) {
      this.tokens.delete(sessionId);
      return false;
    }

    return crypto.timingSafeEqual(
      Buffer.from(storedToken.token),
      Buffer.from(providedToken)
    );
  }

  // Middleware for CSRF protection
  middleware(options: { 
    ignoreMethods?: string[];
    headerName?: string;
    bodyField?: string;
  } = {}) {
    const {
      ignoreMethods = ['GET', 'HEAD', 'OPTIONS'],
      headerName = 'X-CSRF-Token',
      bodyField = '_csrf'
    } = options;

    return (req: Request, res: Response, next: NextFunction) => {
      // Skip CSRF check for safe methods
      if (ignoreMethods.includes(req.method)) {
        return next();
      }

      const sessionId = (req as any).sessionID || req.ip || 'unknown';
      
      // Get token from header or body
      const token = req.get(headerName) || req.body[bodyField];

      if (!token) {
        return res.status(403).json({
          error: 'CSRF token missing',
          message: 'CSRF token is required for this request',
          code: 'csrf_token_missing'
        });
      }

      if (!this.validateToken(sessionId, token)) {
        return res.status(403).json({
          error: 'Invalid CSRF token',
          message: 'The provided CSRF token is invalid or expired',
          code: 'csrf_token_invalid'
        });
      }

      next();
    };
  }

  // Get token for session (for API endpoints)
  getTokenForSession(sessionId: string): string | null {
    const storedToken = this.tokens.get(sessionId);
    
    if (!storedToken || Date.now() > storedToken.expires) {
      return null;
    }

    return storedToken.token;
  }
}

// Create instances
const securityHeaders = new SecurityHeaders();
const csrfProtection = new CSRFProtection();

// Export middleware functions
export const securityHeadersMiddleware = securityHeaders.middleware();
export const csrfMiddleware = csrfProtection.middleware();

// Export instances
export { securityHeaders, csrfProtection };

export default {
  securityHeaders,
  csrfProtection,
  securityHeadersMiddleware,
  csrfMiddleware
};