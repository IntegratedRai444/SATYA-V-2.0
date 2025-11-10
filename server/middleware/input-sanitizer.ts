import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import { validationResult } from 'express-validator';
import { body, query, param } from 'express-validator';

// HTML tags and attributes to remove
const BANNED_TAGS = [
  'script', 'iframe', 'object', 'embed', 'link', 'meta', 'style', 'applet',
  'frame', 'frameset', 'ilayer', 'layer', 'bgsound', 'base', 'xml'
];

const BANNED_ATTRS = [
  'onload', 'onerror', 'onclick', 'onmouseover', 'onmouseout', 'onkeydown',
  'onkeyup', 'onkeypress', 'onfocus', 'onblur', 'onchange', 'onsubmit',
  'javascript:', 'data:', 'vbscript:'
];

// Sanitize input data
const sanitizeInput = (data: any): any => {
  if (typeof data === 'string') {
    // Remove HTML tags
    let sanitized = data.replace(/<\/?[^>]+(>|$)/g, '');
    
    // Remove dangerous attributes
    BANNED_ATTRS.forEach(attr => {
      const regex = new RegExp(attr + '\s*=', 'gi');
      sanitized = sanitized.replace(regex, '');
    });
    
    // Remove control characters
    sanitized = sanitized.replace(/[\x00-\x1F\x7F-\x9F]/g, '');
    
    return sanitized.trim();
  }
  
  if (Array.isArray(data)) {
    return data.map(item => sanitizeInput(item));
  }
  
  if (typeof data === 'object' && data !== null) {
    const sanitized: Record<string, any> = {};
    for (const key in data) {
      sanitized[key] = sanitizeInput(data[key]);
    }
    return sanitized;
  }
  
  return data;
};

// Middleware to sanitize request data
export const sanitizeRequest = (req: Request, res: Response, next: NextFunction) => {
  try {
    // Sanitize request body
    if (req.body && Object.keys(req.body).length > 0) {
      req.body = sanitizeInput(req.body);
    }
    
    // Sanitize query parameters
    if (req.query && Object.keys(req.query).length > 0) {
      req.query = sanitizeInput(req.query);
    }
    
    // Sanitize URL parameters
    if (req.params && Object.keys(req.params).length > 0) {
      req.params = sanitizeInput(req.params);
    }
    
    // Sanitize headers
    if (req.headers) {
      const sensitiveHeaders = ['authorization', 'cookie', 'x-csrf-token'];
      sensitiveHeaders.forEach(header => {
        if (req.headers[header]) {
          // Log redacted version of sensitive headers
          logger.debug(`Sensitive header ${header} received`);
        }
      });
    }
    
    next();
  } catch (error) {
    logger.error('Error in request sanitization:', error);
    next(error);
  }
};

// Common validation rules
export const commonValidationRules = {
  email: body('email')
    .isEmail().withMessage('Please provide a valid email address')
    .normalizeEmail(),
    
  username: body('username')
    .isLength({ min: 3, max: 30 }).withMessage('Username must be between 3 and 30 characters')
    .matches(/^[a-zA-Z0-9_.-]+$/).withMessage('Username can only contain letters, numbers, dots, underscores and hyphens'),
    
  password: body('password')
    .isLength({ min: 8 }).withMessage('Password must be at least 8 characters long')
    .matches(/[A-Z]/).withMessage('Password must contain at least one uppercase letter')
    .matches(/[a-z]/).withMessage('Password must contain at least one lowercase letter')
    .matches(/[0-9]/).withMessage('Password must contain at least one number')
    .matches(/[^A-Za-z0-9]/).withMessage('Password must contain at least one special character'),
    
  idParam: param('id')
    .isMongoId().withMessage('Invalid ID format'),
    
  pagination: [
    query('page')
      .optional()
      .isInt({ min: 1 }).withMessage('Page must be a positive integer')
      .toInt(),
    query('limit')
      .optional()
      .isInt({ min: 1, max: 100 }).withMessage('Limit must be between 1 and 100')
      .toInt()
  ]
};

// Validate request middleware
export const validateRequest = (req: Request, res: Response, next: NextFunction) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      message: 'Validation error',
      errors: errors.array().map(err => ({
        param: err.param,
        message: err.msg,
        location: err.location
      }))
    });
  }
  next();
};

// Export middleware for specific routes
export const validateEmail = [commonValidationRules.email, validateRequest];
export const validateUsername = [commonValidationRules.username, validateRequest];
export const validatePassword = [commonValidationRules.password, validateRequest];
export const validateIdParam = [commonValidationRules.idParam, validateRequest];
export const validatePagination = [...commonValidationRules.pagination, validateRequest];

// Export all validators
export const authValidators = {
  register: [
    commonValidationRules.username,
    commonValidationRules.email,
    commonValidationRules.password,
    validateRequest
  ],
  login: [
    commonValidationRules.email,
    body('password').exists().withMessage('Password is required'),
    validateRequest
  ],
  changePassword: [
    body('currentPassword').exists().withMessage('Current password is required'),
    commonValidationRules.password,
    body('confirmPassword').custom((value, { req }) => {
      if (value !== req.body.newPassword) {
        throw new Error('Passwords do not match');
      }
      return true;
    }),
    validateRequest
  ]
};

// Export all middleware
export default {
  sanitizeRequest,
  validateRequest,
  validators: {
    ...commonValidationRules,
    auth: authValidators,
    validateEmail,
    validateUsername,
    validatePassword,
    validateIdParam,
    validatePagination
  }
};
