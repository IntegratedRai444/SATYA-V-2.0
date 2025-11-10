import { Request, Response, NextFunction, RequestHandler } from 'express';
import { body, param, query, validationResult, ValidationChain } from 'express-validator';
import { isEmail, isURL, isIP, isUUID, isISO8601, isISO31661Alpha2, isPostalCode } from 'validator';
import { Types } from 'mongoose';
import { logger } from '../config/logger';

// Mock audit logger (replace with actual implementation)
const auditLogger = {
  logSuspiciousActivity: async (userId: string, action: string, details: any, severity: string, req: Request) => {
    logger.warn(`[Audit] ${action} - ${severity}`, { userId, details });
  }
};

// Custom types
type ValidationSchema = {
  [key: string]: ValidationChain[];
};

type ValidationRules = {
  [key: string]: ValidationChain | ((field?: string) => ValidationChain);
};

// Custom error formatter
const errorFormatter = ({ msg, param, location, value, nestedErrors }: any) => {
  return { param, location, value, msg };
};

/**
 * Validates the request using express-validator
 * @param schema Validation schema
 */
export const validateRequest = (schema: ValidationSchema): RequestHandler[] => {
  return [
    // 1. Sanitize all inputs
    (req: Request, res: Response, next: NextFunction) => {
      // Create a new sanitized request object with only the data we want to process
      const sanitizedData = {
        body: req.body ? sanitizeData(req.body) : {},
        query: req.query ? sanitizeData(req.query) : {},
        params: req.params ? sanitizeData(req.params) : {}
      };

      // Replace the request properties with sanitized versions
      req.body = sanitizedData.body;
      req.query = sanitizedData.query as any;
      req.params = sanitizedData.params as any;
      
      next();
    },

    // 2. Run validations
    ...Object.entries(schema).flatMap(([location, validations]) => {
      return validations.map(validation => {
        if (location === 'body') return body().custom(validation as any);
        if (location === 'params') return param().custom(validation as any);
        if (location === 'query') return query().custom(validation as any);
        return validation;
      });
    }),

    // 3. Handle validation result
    async (req: Request, res: Response, next: NextFunction) => {
      const errors = validationResult(req).formatWith(errorFormatter);
      if (errors.isEmpty()) {
        return next();
      }

      const errorDetails = errors.array().map((error: any) => ({
        field: error.param || 'unknown',
        message: error.msg,
        value: error.value,
        location: error.location
      }));

      // Log validation failure
      await auditLogger.logSuspiciousActivity(
        (req as any).user?.user_id || 'anonymous',
        'validation_failed',
        {
          endpoint: req.path,
          method: req.method,
          errors: errorDetails,
          ip: req.ip,
          userAgent: req.get('user-agent')
        },
        'medium',
        req
      );

      return res.status(400).json({
        success: false,
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        errors: errorDetails
      });
    }
  ];
};

/**
 * Sanitizes input data to prevent XSS and injection attacks
 */
const sanitizeData = (data: any): any => {
  if (data === null || data === undefined) {
    return data;
  }

  // Handle strings
  if (typeof data === 'string') {
    // Remove null bytes and other control characters
    let sanitized = data
      .replace(/\0/g, '')
      // Remove HTML tags
      .replace(/<[^>]*>?/gm, '')
      // Remove potentially dangerous characters
      .replace(/[\u0000-\u001F\u007F-\u009F\u2000-\u200F\u2028-\u202F\u205F-\u206F\u3000\uFEFF]/g, '')
      // Remove SQL injection patterns
      .replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b|;|--|\/\*|\*\/|xp_|sp_)/gi, '')
      // Trim whitespace
      .trim();

    // Additional sanitization for specific data types
    if (data.match(/^[0-9a-fA-F]{24}$/)) {
      // MongoDB ObjectID
      return Types.ObjectId.isValid(data) ? sanitized : '';
    }

    return sanitized;
  }
  
  // Handle arrays
  if (Array.isArray(data)) {
    return data.map(item => sanitizeData(item));
  }
  
  // Handle objects
  if (typeof data === 'object') {
    const sanitized: Record<string, any> = {};
    for (const key in data) {
      if (Object.prototype.hasOwnProperty.call(data, key)) {
        sanitized[sanitizeData(key)] = sanitizeData(data[key]);
      }
    }
    return sanitized;
  }

  return data;
};

/**
 * Common validation rules
 */
export const v = {
  // Field validators
  required: (field = 'field') => 
    body(field).notEmpty().withMessage(`${field} is required`),
  
  optional: (field: string) => 
    body(field).optional(),
  
  // Type validators
  string: (field = 'field') => 
    body(field).isString().withMessage(`${field} must be a string`),
    
  number: (field = 'field') => 
    body(field).isNumeric().withMessage(`${field} must be a number`),
    
  boolean: (field = 'field') => 
    body(field).isBoolean().withMessage(`${field} must be a boolean`),
    
  array: (field = 'field') => 
    body(field).isArray().withMessage(`${field} must be an array`),
    
  object: (field = 'field') => 
    body(field).isObject().withMessage(`${field} must be an object`),
    
  // Format validators
  email: (field = 'email') => 
    body(field)
      .trim()
      .isEmail()
      .withMessage('Invalid email format')
      .normalizeEmail(),
      
  password: (field = 'password') => 
    body(field)
      .isLength({ min: 8 })
      .withMessage('Password must be at least 8 characters')
      .matches(/[A-Z]/)
      .withMessage('Password must contain at least one uppercase letter')
      .matches(/[a-z]/)
      .withMessage('Password must contain at least one lowercase letter')
      .matches(/[0-9]/)
      .withMessage('Password must contain at least one number')
      .matches(/[^A-Za-z0-9]/)
      .withMessage('Password must contain at least one special character'),
      
  url: (field = 'url') =>
    body(field)
      .custom((value) => isURL(value, { 
        protocols: ['http', 'https'], 
        require_protocol: true,
        require_valid_protocol: true 
      }))
      .withMessage('Invalid URL format'),
      
  ip: (field = 'ip') =>
    body(field)
      .custom((value) => isIP(value))
      .withMessage('Invalid IP address'),
      
  uuid: (field = 'id') =>
    body(field)
      .isUUID()
      .withMessage('Invalid UUID format'),
      
  date: (field = 'date') =>
    body(field)
      .custom((value) => isISO8601(value))
      .withMessage('Invalid date format. Use ISO 8601 format'),
      
  countryCode: (field = 'countryCode') =>
    body(field)
      .custom((value) => isISO31661Alpha2(value))
      .withMessage('Invalid country code. Use ISO 3166-1 alpha-2 format'),
          
  postalCode: (field = 'postalCode', countryCode: any = 'US') =>
    body(field)
      .custom((value) => {
        try {
          // Cast to any to bypass TypeScript type checking for the second parameter
          return isPostalCode(value, countryCode as any);
        } catch (error) {
          return false;
        }
      })
      .withMessage(`Invalid postal code format for ${countryCode}`),
      
  // Length validators
  minLength: (field = 'field', length: number) =>
    body(field)
      .isLength({ min: length })
      .withMessage(`${field} must be at least ${length} characters`),
      
  maxLength: (field = 'field', length: number) =>
    body(field)
      .isLength({ max: length })
      .withMessage(`${field} must be at most ${length} characters`),
      
  length: (field = 'field', min: number, max: number) =>
    body(field)
      .isLength({ min, max })
      .withMessage(`${field} must be between ${min} and ${max} characters`),
      
  // Numeric validators
  min: (field = 'field', min: number) =>
    body(field)
      .isFloat({ min })
      .withMessage(`${field} must be at least ${min}`),
      
  max: (field = 'field', max: number) =>
    body(field)
      .isFloat({ max })
      .withMessage(`${field} must be at most ${max}`),
      
  range: (field = 'field', min: number, max: number) =>
    body(field)
      .isFloat({ min, max })
      .withMessage(`${field} must be between ${min} and ${max}`),
      
  // Array validators
  arrayMin: (field = 'field', min: number) =>
    body(field)
      .isArray({ min })
      .withMessage(`${field} must contain at least ${min} items`),
      
  arrayMax: (field = 'field', max: number) =>
    body(field)
      .isArray({ max })
      .withMessage(`${field} must contain at most ${max} items`),
      
  arrayLength: (field = 'field', length: number) =>
    body(field)
      .isArray({ min: length, max: length })
      .withMessage(`${field} must contain exactly ${length} items`),
      
  // ObjectId validator
  objectId: (field = 'id') =>
    body(field)
      .custom((value) => Types.ObjectId.isValid(value))
      .withMessage('Invalid ID format'),
      
  // Enum validator
  enum: (field = 'field', values: any[]) =>
    body(field)
      .isIn(values)
      .withMessage(`Invalid value. Must be one of: ${values.join(', ')}`),
      
  // Custom validator
  custom: (field: string, validator: (value: any) => boolean, message: string) =>
    body(field)
      .custom(validator)
      .withMessage(message)
};

// Common validation schemas
export const schemas = {
  // User validation
  login: [
    v.required('username'),
    v.string('username'),
    v.minLength('username', 3),
    v.maxLength('username', 50),
    v.required('password'),
    v.string('password'),
    v.minLength('password', 8),
    v.maxLength('password', 128),
  ],
  // File upload validation
  fileUpload: [
    v.optional('fileName'),
    v.string('fileName'),
    v.maxLength('fileName', 255),
    v.optional('fileSize'),
    v.number('fileSize'),
    v.min('fileSize', 1),
    v.max('fileSize', 100 * 1024 * 1024), // 100MB max
  ],
  // Analysis parameters
  analysisParams: [
    v.optional('analysisType'),
    v.enum('analysisType', ['image', 'video', 'audio', 'multimodal']),
    v.optional('options.quality'),
    v.enum('options.quality', ['fast', 'balanced', 'thorough']),
    v.optional('options.includeMetadata'),
    v.boolean('options.includeMetadata'),
  ],
  // Session validation
  sessionToken: [
    v.required('token'),
    v.string('token'),
    v.minLength('token', 10),
    v.maxLength('token', 500),
  ],
  // Query parameters
  pagination: [
    v.optional('page'),
    v.number('page'),
    v.min('page', 1),
    v.max('page', 1000),
    v.optional('limit'),
    v.number('limit'),
    v.min('limit', 1),
    v.max('limit', 100),
  ],
  // ID parameters
  objectId: [
    v.required('id'),
    v.string('id'),
    v.minLength('id', 1),
    v.maxLength('id', 50),
  ],
  // Search parameters
  search: [
    v.optional('q'),
    v.string('q'),
    v.maxLength('q', 200),
  ]
};

/**
 * File upload validation utilities
 */
const fileUploadValidation = {
  // Validate file type
  validateFileType: (allowedTypes: string[]) => {
    return (req: Request, res: Response, next: NextFunction) => {
      if (!req.file && !req.files) {
        return next();
      }

      const files = req.files 
        ? (Array.isArray(req.files) 
            ? req.files 
            : Object.values(req.files).flat()
          ) 
        : [req.file];
      
      for (const file of files) {
        if (!file) continue;
        
        const fileExtension = file.originalname?.split('.').pop()?.toLowerCase() || '';
        const mimeType = file.mimetype?.toLowerCase() || '';
        
        const isValidExtension = allowedTypes.some(type => 
          fileExtension.includes(type.toLowerCase())
        );
        
        const isValidMimeType = allowedTypes.some(type => 
          mimeType.includes(type.toLowerCase())
        );

        if (!isValidExtension && !isValidMimeType) {
          return res.status(400).json({
            success: false,
            message: `Invalid file type. Allowed types: ${allowedTypes.join(', ')}`,
            code: 'INVALID_FILE_TYPE',
            allowedTypes
          });
        }
      }

      next();
    };
  },

  // Validate file size
  validateFileSize: (maxSizeBytes: number) => {
    return (req: Request, res: Response, next: NextFunction) => {
      if (!req.file && !req.files) {
        return next();
      }

      const files = req.files 
        ? (Array.isArray(req.files) 
            ? req.files 
            : Object.values(req.files).flat()
          ) 
        : [req.file];
      
      for (const file of files) {
        if (!file) continue;
        
        if (file.size > maxSizeBytes) {
          const maxSizeMB = Math.round(maxSizeBytes / (1024 * 1024));
          return res.status(400).json({
            success: false,
            message: `File too large. Maximum size: ${maxSizeMB}MB`,
            code: 'FILE_TOO_LARGE',
            maxSize: maxSizeBytes,
            fileSize: file.size
          });
        }
      }

      next();
    };
  }
};

// File size validation
export const validateFileSize = (maxSize: number) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.file && !req.files) {
      return next();
    }

    const files = req.files ? (Array.isArray(req.files) ? req.files : Object.values(req.files).flat()) : [req.file];
    
    for (const file of files) {
      if (!file) continue;
      
      if (file.size > maxSize) {
        return res.status(400).json({
          error: 'File too large',
          message: `File size exceeds maximum allowed size of ${Math.round(maxSize / (1024 * 1024))}MB`,
          code: 'file_too_large',
          maxSize,
          fileSize: file.size
        });
      }
    }

    next();
  };
};

// Content-Type validation
export const validateContentType = (allowedTypes: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const contentType = req.get('Content-Type');
    
    if (!contentType) {
      return res.status(400).json({
        error: 'Missing Content-Type header',
        code: 'missing_content_type',
        allowedTypes
      });
    }

    const isValid = allowedTypes.some(type => contentType.includes(type));
    
    if (!isValid) {
      return res.status(400).json({
        error: 'Invalid Content-Type',
        message: `Content-Type not allowed. Allowed types: ${allowedTypes.join(', ')}`,
        code: 'invalid_content_type',
        allowedTypes,
        receivedType: contentType
      });
    }

    next();
  };
};

// SQL injection prevention
const security = {
  // Prevent SQL injection
  preventSQLInjection: (req: Request, res: Response, next: NextFunction) => {
    const sqlPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/gi,
      /(;|--|\/\*|\*\/|xp_|sp_)/gi,
      /(\b(OR|AND)\b.*=.*)/gi
    ];

    const checkForSQLInjection = (obj: any): boolean => {
      if (typeof obj === 'string') {
        return sqlPatterns.some(pattern => pattern.test(obj));
      }
      
      if (Array.isArray(obj)) {
        return obj.some(item => checkForSQLInjection(item));
      }
      
      if (obj && typeof obj === 'object') {
        return Object.values(obj).some(value => checkForSQLInjection(value));
      }
      
      return false;
    };

    if (checkForSQLInjection(req.body) || checkForSQLInjection(req.query) || checkForSQLInjection(req.params)) {
      auditLogger.logSuspiciousActivity(
        (req as any).user?.user_id || 'anonymous',
        'sql_injection_attempt',
        {
          endpoint: req.path,
          method: req.method,
          ip: req.ip,
          userAgent: req.get('user-agent')
        },
        'critical',
        req
      );

      return res.status(400).json({
        success: false,
        message: 'Invalid input detected',
        code: 'SECURITY_VIOLATION'
      });
    }

    next();
  },

  // Prevent XSS attacks
  preventXSS: (req: Request, res: Response, next: NextFunction) => {
    const xssPatterns = [
      /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi,
      /<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/gi
    ];

    const checkForXSS = (obj: any): boolean => {
      if (typeof obj === 'string') {
        return xssPatterns.some(pattern => pattern.test(obj));
      }
      
      if (Array.isArray(obj)) {
        return obj.some(item => checkForXSS(item));
      }
      
      if (obj && typeof obj === 'object') {
        return Object.values(obj).some(value => checkForXSS(value));
      }
      
      return false;
    };

    if (checkForXSS(req.body) || checkForXSS(req.query) || checkForXSS(req.params)) {
      auditLogger.logSuspiciousActivity(
        (req as any).user?.user_id || 'anonymous',
        'xss_attempt',
        {
          endpoint: req.path,
          method: req.method,
          ip: req.ip,
          userAgent: req.get('user-agent')
        },
        'high',
        req
      );

      return res.status(400).json({
        success: false,
        message: 'Potentially malicious input detected',
        code: 'XSS_ATTEMPT'
      });
    }

    next();
  }
};

// Export all validators and middleware
export default {
  // Core validation
  validateRequest,
  
  // Validation helpers
  v,
  schemas,
  
  // File upload validation
  fileUpload: fileUploadValidation,
  
  // Security middleware
  security,
  
  // Error handling
  handleValidationErrors: async (req: Request, res: Response, next: NextFunction) => {
    const errors = validationResult(req);
    
    if (!errors.isEmpty()) {
      const errorDetails = errors.array().map((error: any) => ({
        field: error.path || 'unknown',
        message: error.msg,
        value: error.value,
        location: error.location
      }));

      // Log validation failure
      await auditLogger.logSuspiciousActivity(
        (req as any).user?.user_id || 'anonymous',
        'validation_failed',
        {
          endpoint: req.path,
          method: req.method,
          errors: errorDetails,
          ip: req.ip,
          userAgent: req.get('user-agent')
        },
        'medium',
        req
      );

      return res.status(400).json({
        success: false,
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        errors: errorDetails
      });
    }
    
    next();
  }
};