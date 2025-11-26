import type { Request, Response, NextFunction } from 'express';
import { config } from '../config';

/**
 * Cookie configuration for JWT tokens
 */
export const COOKIE_CONFIG = {
  name: 'satyaai_token',
  options: {
    httpOnly: true,           // Prevents JavaScript access (XSS protection)
    secure: config.NODE_ENV === 'production', // HTTPS only in production
    sameSite: 'strict' as const, // CSRF protection
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    path: '/',
    domain: config.NODE_ENV === 'production' ? undefined : 'localhost'
  }
};

/**
 * Set JWT token as httpOnly cookie
 */
export function setTokenCookie(res: Response, token: string): void {
  res.cookie(COOKIE_CONFIG.name, token, COOKIE_CONFIG.options);
}

/**
 * Clear JWT token cookie
 */
export function clearTokenCookie(res: Response): void {
  res.clearCookie(COOKIE_CONFIG.name, {
    httpOnly: true,
    secure: config.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
    domain: config.NODE_ENV === 'production' ? undefined : 'localhost'
  });
}

/**
 * Extract token from request (supports both cookie and Authorization header)
 * Priority: Authorization header > Cookie
 */
export function extractToken(req: Request): string | null {
  // First, check Authorization header (for API clients and mobile apps)
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }

  // Second, check httpOnly cookie (for web browsers)
  if (req.cookies && req.cookies[COOKIE_CONFIG.name]) {
    return req.cookies[COOKIE_CONFIG.name];
  }

  return null;
}

/**
 * Middleware to parse cookies (if not already using cookie-parser)
 */
export function parseCookies(req: Request, res: Response, next: NextFunction): void {
  if (!req.cookies) {
    const cookieHeader = req.headers.cookie;
    if (cookieHeader) {
      req.cookies = {};
      cookieHeader.split(';').forEach(cookie => {
        const [name, ...rest] = cookie.split('=');
        const value = rest.join('=').trim();
        if (name && value) {
          req.cookies[name.trim()] = decodeURIComponent(value);
        }
      });
    } else {
      req.cookies = {};
    }
  }
  next();
}

// Extend Express Request type to include cookies
declare global {
  namespace Express {
    interface Request {
      cookies?: { [key: string]: string };
    }
  }
}
