import type { Request, Response, NextFunction, RequestHandler } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../config/logger';
import { getSecurityConfig } from '../config/security';

// Extend the Express Request type to include session
declare global {
  namespace Express {
    interface Request {
      session?: {
        csrfToken?: string;
        [key: string]: any;
      };
      csrfToken?: () => string;
    }
  }
}

const securityConfig = getSecurityConfig();

/**
 * CSRF Protection Middleware
 * Implements the double submit cookie pattern for CSRF protection
 */
export const csrfProtection = (req: Request, res: Response, next: NextFunction) => {
  // Skip CSRF check for safe methods
  if (['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(req.method)) {
    return next();
  }

  // Skip CSRF check if disabled in config
  if (!securityConfig.csrf.enabled) {
    return next();
  }

  // Get tokens from request
  const csrfToken = req.headers[securityConfig.csrf.headerName.toLowerCase()] || 
                   req.body[securityConfig.csrf.fieldName];
  const csrfCookie = req.cookies ? req.cookies[securityConfig.csrf.cookie.name] : null;

  // Verify CSRF token
  if (!csrfToken || !csrfCookie || csrfToken !== csrfCookie) {
    logger.warn('CSRF token validation failed', {
      ip: req.ip,
      method: req.method,
      path: req.path,
      hasToken: !!csrfToken,
      hasCookie: !!csrfCookie,
      tokenMatch: csrfToken === csrfCookie
    });

    return res.status(403).json({
      success: false,
      message: 'Invalid or missing CSRF token',
      code: 'INVALID_CSRF_TOKEN'
    });
  }

  next();
};

/**
 * Middleware to generate and set CSRF token
 */
export const csrfToken = (req: Request, res: Response, next: NextFunction) => {
  if (!securityConfig.csrf.enabled) {
    return next();
  }

  // Generate a new CSRF token if one doesn't exist
  let token = req.cookies ? req.cookies['XSRF-TOKEN'] : null;
  if (!token) {
    token = uuidv4();
    
    // Set the CSRF token as an HTTP-only cookie
    const cookieOptions = {
      httpOnly: securityConfig.csrf.cookie.httpOnly,
      secure: securityConfig.csrf.cookie.secure,
      sameSite: securityConfig.csrf.cookie.sameSite as 'strict' | 'lax' | 'none',
      maxAge: securityConfig.csrf.cookie.maxAge,
      path: securityConfig.csrf.cookie.path || '/',
      domain: securityConfig.csrf.cookie.domain
    };
    
    // Remove undefined properties
    Object.keys(cookieOptions).forEach(key => 
      cookieOptions[key as keyof typeof cookieOptions] === undefined && 
      delete cookieOptions[key as keyof typeof cookieOptions]
    );
    
    res.cookie(securityConfig.csrf.cookie.name, token, cookieOptions);
  }

  // Make the token available to views
  res.locals.csrfToken = token;
  next();
};

/**
 * Middleware to verify CSRF token for API requests
 */
export const verifyCsrfToken = (req: Request, res: Response, next: NextFunction) => {
  // Skip CSRF check for safe methods
  if (['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(req.method)) {
    return next();
  }

  // Skip CSRF check if disabled in config
  if (!securityConfig.csrf.enabled) {
    return next();
  }

  // Get the CSRF token from the request
  const csrfToken = req.headers[securityConfig.csrf.headerName.toLowerCase()] || 
                   req.body[securityConfig.csrf.fieldName];
  
  // Verify the CSRF token
  const sessionToken = req.session?.csrfToken;
  const isValidToken = csrfToken && sessionToken && csrfToken === sessionToken;
  
  if (!isValidToken) {
    logger.warn('CSRF token validation failed', {
      ip: req.ip,
      method: req.method,
      path: req.path,
      hasToken: !!csrfToken,
      sessionToken: req.session?.csrfToken ? 'present' : 'missing'
    });

    return res.status(403).json({
      success: false,
      message: 'Invalid or missing CSRF token',
      code: 'INVALID_CSRF_TOKEN'
    });
  }

  next();
};

// Type for the CSRF middleware
export type CsrfMiddleware = {
  csrfProtection: RequestHandler;
  csrfToken: RequestHandler;
  verifyCsrfToken: RequestHandler;
};

// Export the CSRF middleware
export const csrfMiddleware: CsrfMiddleware = {
  csrfProtection,
  csrfToken,
  verifyCsrfToken
};

export default csrfMiddleware;
