import type { Request, Response, NextFunction } from 'express';
import DOMPurify from 'isomorphic-dompurify';
import { logger } from '../config/logger';

/**
 * Input Sanitization Middleware
 * Sanitizes all string inputs to prevent XSS and injection attacks
 */
export const sanitizeInput = (req: Request, res: Response, next: NextFunction) => {
  try {
    // Sanitize request body
    if (req.body && typeof req.body === 'object') {
      req.body = sanitizeObject(req.body);
    }

    // Sanitize query parameters
    if (req.query && typeof req.query === 'object') {
      req.query = sanitizeObject(req.query);
    }

    // Sanitize URL parameters
    if (req.params && typeof req.params === 'object') {
      req.params = sanitizeObject(req.params);
    }

    next();
  } catch (error) {
    logger.error('Input sanitization error:', error);
    next(error);
  }
};

/**
 * Recursively sanitize an object by cleaning all string values
 */
function sanitizeObject(obj: any): any {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(sanitizeObject);
  }

  const sanitized: any = {};
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      sanitized[key] = DOMPurify.sanitize(value, {
        ALLOWED_TAGS: [],
        ALLOWED_ATTR: []
      });
    } else if (typeof value === 'object') {
      sanitized[key] = sanitizeObject(value);
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}

/**
 * Validate and sanitize email addresses
 */
export const sanitizeEmail = (email: string): string => {
  if (!email) return '';
  
  // Basic email sanitization
  const sanitized = email.toLowerCase().trim();
  
  // Remove potentially dangerous characters
  return sanitized.replace(/[<>]/g, '');
};

/**
 * Validate and sanitize phone numbers
 */
export const sanitizePhone = (phone: string): string => {
  if (!phone) return '';
  
  // Remove all non-numeric characters except + and -
  return phone.replace(/[^\d\+\-]/g, '');
};

/**
 * Validate and sanitize URLs
 */
export const sanitizeUrl = (url: string): string => {
  if (!url) return '';
  
  try {
    // Basic URL validation and sanitization
    const sanitized = url.trim();
    
    // Only allow http, https protocols
    if (!sanitized.startsWith('http://') && !sanitized.startsWith('https://')) {
      return '';
    }
    
    // Remove potentially dangerous characters
    return sanitized.replace(/[<>"'\\]/g, '');
  } catch {
    return '';
  }
};

/**
 * Validate and sanitize file names
 */
export const sanitizeFileName = (fileName: string): string => {
  if (!fileName) return '';
  
  // Remove path traversal attempts and dangerous characters
  return fileName
    .replace(/[\\\/]/g, '_')
    .replace(/\.\./g, '')
    .replace(/[<>:"|?*]/g, '_')
    .toLowerCase();
};

export default {
  sanitizeInput,
  sanitizeEmail,
  sanitizePhone,
  sanitizeUrl,
  sanitizeFileName
};
