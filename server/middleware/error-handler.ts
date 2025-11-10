import { Request, Response, NextFunction } from 'express';
import { ZodError } from 'zod';
import multer from 'multer';
import { logger, config } from '../config';

export interface AppError extends Error {
  statusCode?: number;
  code?: string;
  isOperational?: boolean;
  details?: any;
}

export class CustomError extends Error implements AppError {
  public statusCode: number;
  public code: string;
  public isOperational: boolean;
  public details?: any;

  constructor(
    message: string,
    statusCode: number = 500,
    code: string = 'INTERNAL_ERROR',
    isOperational: boolean = true,
    details?: any
  ) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = isOperational;
    this.details = details;

    Error.captureStackTrace(this, this.constructor);
  }
}

// Predefined error classes
export class ValidationError extends CustomError {
  constructor(message: string, details?: any) {
    super(message, 400, 'VALIDATION_ERROR', true, details);
  }
}

export class AuthenticationError extends CustomError {
  constructor(message: string = 'Authentication required') {
    super(message, 401, 'AUTHENTICATION_ERROR', true);
  }
}

export class AuthorizationError extends CustomError {
  constructor(message: string = 'Access denied') {
    super(message, 403, 'AUTHORIZATION_ERROR', true);
  }
}

export class NotFoundError extends CustomError {
  constructor(message: string = 'Resource not found') {
    super(message, 404, 'NOT_FOUND_ERROR', true);
  }
}

export class ConflictError extends CustomError {
  constructor(message: string = 'Resource conflict') {
    super(message, 409, 'CONFLICT_ERROR', true);
  }
}

export class RateLimitError extends CustomError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message, 429, 'RATE_LIMIT_ERROR', true);
  }
}

export class ServiceUnavailableError extends CustomError {
  constructor(message: string = 'Service temporarily unavailable') {
    super(message, 503, 'SERVICE_UNAVAILABLE_ERROR', true);
  }
}

export class FileProcessingError extends CustomError {
  constructor(message: string, details?: any) {
    super(message, 422, 'FILE_PROCESSING_ERROR', true, details);
  }
}

export class AnalysisError extends CustomError {
  constructor(message: string, details?: any) {
    super(message, 500, 'ANALYSIS_ERROR', true, details);
  }
}

/**
 * Format error response
 */
function formatErrorResponse(error: AppError, req: Request): {
  success: boolean;
  error: string;
  message: string;
  code: string;
  timestamp: string;
  path: string;
  method: string;
  details?: any;
  stack?: string;
  requestId?: string;
} {
  const response = {
    success: false,
    error: error.name || 'Error',
    message: error.message,
    code: error.code || 'UNKNOWN_ERROR',
    timestamp: new Date().toISOString(),
    path: req.path,
    method: req.method,
    requestId: req.headers['x-request-id'] as string || undefined
  };

  // Add details in development or for operational errors
  if (config.NODE_ENV !== 'production' || error.isOperational) {
    if (error.details) {
      (response as any).details = error.details;
    }
  }

  // Add stack trace in development
  if (config.NODE_ENV !== 'production') {
    (response as any).stack = error.stack;
  }

  return response;
}

/**
 * Log error with context
 */
function logError(error: AppError, req: Request, context?: any): void {
  const logLevel = error.statusCode && error.statusCode < 500 ? 'warn' : 'error';
  
  const logData = {
    error: {
      name: error.name,
      message: error.message,
      code: error.code,
      statusCode: error.statusCode,
      stack: error.stack
    },
    request: {
      method: req.method,
      path: req.path,
      query: req.query,
      params: req.params,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      userId: (req as any).user?.userId,
      requestId: req.headers['x-request-id']
    },
    context
  };

  if (logLevel === 'error') {
    logger.error('Request error', logData);
  } else {
    logger.warn('Request warning', logData);
  }
}

/**
 * Handle Zod validation errors
 */
function handleZodError(error: ZodError): ValidationError {
  const details = error.errors.map(err => ({
    field: err.path.join('.'),
    message: err.message,
    code: err.code,
    received: err.received
  }));

  return new ValidationError(
    'Validation failed',
    { validationErrors: details }
  );
}

/**
 * Handle Multer file upload errors
 */
function handleMulterError(error: multer.MulterError): CustomError {
  let message = 'File upload error';
  let code = 'FILE_UPLOAD_ERROR';

  switch (error.code) {
    case 'LIMIT_FILE_SIZE':
      message = 'File size exceeds maximum allowed size';
      code = 'FILE_TOO_LARGE';
      break;
    case 'LIMIT_FILE_COUNT':
      message = 'Too many files uploaded';
      code = 'TOO_MANY_FILES';
      break;
    case 'LIMIT_UNEXPECTED_FILE':
      message = 'Unexpected file field';
      code = 'UNEXPECTED_FILE';
      break;
    case 'LIMIT_PART_COUNT':
      message = 'Too many parts in multipart form';
      code = 'TOO_MANY_PARTS';
      break;
    case 'LIMIT_FIELD_KEY':
      message = 'Field name too long';
      code = 'FIELD_NAME_TOO_LONG';
      break;
    case 'LIMIT_FIELD_VALUE':
      message = 'Field value too long';
      code = 'FIELD_VALUE_TOO_LONG';
      break;
    case 'LIMIT_FIELD_COUNT':
      message = 'Too many fields';
      code = 'TOO_MANY_FIELDS';
      break;
  }

  return new ValidationError(message, { multerCode: error.code, field: error.field });
}

/**
 * Handle database errors
 */
function handleDatabaseError(error: any): CustomError {
  // SQLite specific errors
  if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') {
    return new ConflictError('Resource already exists');
  }
  
  if (error.code === 'SQLITE_CONSTRAINT_FOREIGNKEY') {
    return new ValidationError('Invalid reference to related resource');
  }

  if (error.code === 'SQLITE_CONSTRAINT_NOTNULL') {
    return new ValidationError('Required field is missing');
  }

  // Generic database error
  return new CustomError(
    'Database operation failed',
    500,
    'DATABASE_ERROR',
    true,
    { originalError: error.message }
  );
}

/**
 * Handle network/HTTP errors
 */
function handleNetworkError(error: any): CustomError {
  if (error.code === 'ECONNREFUSED') {
    return new ServiceUnavailableError('External service is unavailable');
  }

  if (error.code === 'ETIMEDOUT') {
    return new ServiceUnavailableError('Request timeout');
  }

  if (error.response) {
    // HTTP error response
    const status = error.response.status;
    const message = error.response.data?.message || 'External service error';
    
    if (status >= 500) {
      return new ServiceUnavailableError(message);
    } else if (status === 404) {
      return new NotFoundError(message);
    } else if (status === 401) {
      return new AuthenticationError(message);
    } else if (status === 403) {
      return new AuthorizationError(message);
    } else {
      return new ValidationError(message);
    }
  }

  return new ServiceUnavailableError('Network error occurred');
}

/**
 * Convert unknown errors to AppError
 */
function normalizeError(error: any): AppError {
  // Already an AppError
  if (error instanceof CustomError) {
    return error;
  }

  // Zod validation error
  if (error instanceof ZodError) {
    return handleZodError(error);
  }

  // Multer file upload error
  if (error instanceof multer.MulterError) {
    return handleMulterError(error);
  }

  // Database errors
  if (error.code && error.code.startsWith('SQLITE_')) {
    return handleDatabaseError(error);
  }

  // Network/HTTP errors
  if (error.code || error.response) {
    return handleNetworkError(error);
  }

  // JWT errors
  if (error.name === 'JsonWebTokenError') {
    return new AuthenticationError('Invalid token');
  }

  if (error.name === 'TokenExpiredError') {
    return new AuthenticationError('Token expired');
  }

  // Generic error
  const appError = new CustomError(
    error.message || 'An unexpected error occurred',
    error.statusCode || 500,
    error.code || 'INTERNAL_ERROR',
    false // Not operational since we don't know what it is
  );

  // Preserve original stack trace
  appError.stack = error.stack;

  return appError;
}

/**
 * Main error handling middleware
 */
export function errorHandler(
  error: any,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Normalize error
  const appError = normalizeError(error);

  // Log error
  logError(appError, req);

  // Don't send error response if headers already sent
  if (res.headersSent) {
    return next(error);
  }

  // Format and send error response
  const errorResponse = formatErrorResponse(appError, req);
  res.status(appError.statusCode || 500).json(errorResponse);
}

/**
 * Handle 404 errors (no route matched)
 */
export function notFoundHandler(req: Request, res: Response): void {
  const error = new NotFoundError(`Route ${req.method} ${req.path} not found`);
  const errorResponse = formatErrorResponse(error, req);
  
  logger.warn('Route not found', {
    method: req.method,
    path: req.path,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  res.status(404).json(errorResponse);
}

/**
 * Async error wrapper for route handlers
 */
export function asyncHandler<T extends Request = Request>(
  fn: (req: T, res: Response, next: NextFunction) => Promise<any>
) {
  return (req: T, res: Response, next: NextFunction): void => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

/**
 * Create error with context
 */
export function createError(
  message: string,
  statusCode: number = 500,
  code: string = 'INTERNAL_ERROR',
  details?: any
): CustomError {
  return new CustomError(message, statusCode, code, true, details);
}

/**
 * Assert condition or throw error
 */
export function assert(
  condition: any,
  message: string,
  statusCode: number = 400,
  code: string = 'ASSERTION_ERROR'
): asserts condition {
  if (!condition) {
    throw new CustomError(message, statusCode, code, true);
  }
}

/**
 * Middleware to add request ID
 */
export function requestIdMiddleware(req: Request, res: Response, next: NextFunction): void {
  const requestId = req.headers['x-request-id'] as string || 
                   `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  req.headers['x-request-id'] = requestId;
  res.setHeader('X-Request-ID', requestId);
  
  next();
}

/**
 * Graceful shutdown handler
 */
export function setupGracefulShutdown(): void {
  const gracefulShutdown = (signal: string) => {
    logger.info(`Received ${signal}, starting graceful shutdown...`);
    
    // Give ongoing requests time to complete
    setTimeout(() => {
      logger.info('Graceful shutdown completed');
      process.exit(0);
    }, 10000); // 10 seconds
  };

  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
  process.on('SIGINT', () => gracefulShutdown('SIGINT'));

  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', {
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      }
    });
    
    // Exit after logging
    setTimeout(() => process.exit(1), 1000);
  });

  // Handle unhandled promise rejections
  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection', {
      reason: reason,
      promise: promise
    });
    
    // Exit after logging
    setTimeout(() => process.exit(1), 1000);
  });
}