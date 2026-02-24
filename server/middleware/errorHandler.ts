import { Request, Response, NextFunction, RequestHandler } from 'express';
import { validationResult, ValidationChain } from 'express-validator';
import { errorResponse, serverError, ApiError } from '../utils/apiResponse';
import { logger } from '../config/logger';

/**
 * Validates the request using express-validator
 */
export const validateRequest = (validations: ValidationChain[]): RequestHandler => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return async (req: any, res: Response, next: NextFunction) => {
    await Promise.all(validations.map(validation => validation.run(req)));

    const errors = validationResult(req);
    if (errors.isEmpty()) {
      return next();
    }

    const errorMessages = errors.array().map(err => {
      // Handle both standard and alternative validation errors
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const errorObj: any = {
        message: err.msg,
      };
      
      // Only add field and value if they exist
      if ('param' in err && err.param) {
        errorObj.field = err.param;
      }
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ('value' in (err as any) && (err as any).value !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        errorObj.value = (err as any).value;
      }
      
      return errorObj;
    });

    return errorResponse(
      res,
      {
        code: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: errorMessages,
      },
      400
    );
  };
};

/**
 * Global error handler middleware
 */
interface ErrorWithStatus extends Error {
  status?: number;
  statusCode?: number;
  code?: string; // Ensure code is always a string for API responses
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  details?: any;
  expose?: boolean;
  retryAfter?: number;
  msBeforeNext?: number;
}

export const errorHandler = (
  err: ErrorWithStatus,
  req: Request,
  res: Response,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  next: NextFunction
) => {
  // Log the error
  logger.error('Error handler:', {
    error: {
      message: err.message,
      stack: err.stack,
      name: err.name,
    },
    request: {
      method: req.method,
      url: req.originalUrl,
      params: req.params,
      query: req.query,
      body: req.body,
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    requestId: (req as any).requestId,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    traceId: (req as any).traceId,
  });

  // Handle JWT errors
  if (err.name === 'JsonWebTokenError') {
    return errorResponse(
      res,
      {
        code: 'INVALID_TOKEN',
        message: 'Invalid token',
      },
      401
    );
  }

  if (err.name === 'TokenExpiredError') {
    return errorResponse(
      res,
      {
        code: 'TOKEN_EXPIRED',
        message: 'Token has expired',
      },
      401
    );
  }

  // Handle validation errors
  if (err.name === 'ValidationError' || err.name === 'ValidatorError') {
    const details = err.details || err.message;
    
    // Handle different validation error formats
    const formattedDetails = Array.isArray(details)
      ? details
      : typeof details === 'string'
      ? [{ message: details }]
      : [{ message: 'Validation failed' }];

    return errorResponse(
      res,
      {
        code: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: formattedDetails,
      },
      422
    );
  }

  // Handle 404 errors
  if (err.status === 404) {
    return errorResponse(
      res,
      {
        code: 'NOT_FOUND',
        message: err.message || 'Resource not found',
      },
      404
    );
  }

  // Handle rate limit errors
  if (err.status === 429) {
    const retryAfter = err.retryAfter || Math.ceil((err.msBeforeNext || 60000) / 1000);
    
    return errorResponse(
      res,
      {
        code: 'RATE_LIMIT_EXCEEDED',
        message: 'Too many requests, please try again later',
        retryAfter,
      },
      429
    );
  }

  // Handle API errors
  if (err instanceof ApiError) {
    return errorResponse(
      res,
      {
        code: err.code,
        message: err.message,
        details: err.details,
        ...(err.retryAfter && { retryAfter: err.retryAfter }),
      },
      err.statusCode
    );
  }

  // Handle other known error types
  if (err.status && err.status >= 400 && err.status < 500) {
    return errorResponse(
      res,
      {
        code: err.code || 'BAD_REQUEST',
        message: err.message || 'Bad request',
        details: err.details,
        ...(err.retryAfter && { retryAfter: err.retryAfter }),
      },
      err.status
    );
  }

  // Fallback to generic server error
  return serverError(res, err.message || 'Internal server error');
};

/**
 * Wraps async route handlers to ensure errors are caught and passed to next()
 */
export const asyncHandler = <T extends RequestHandler>(
  fn: T
): RequestHandler => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (req: any, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch((error) => {
      // Log the error with request context
      logger.error('Async handler error', {
        error: error.message,
        stack: error.stack,
        path: req.path,
        method: req.method,
        params: req.params,
        query: req.query,
        // Don't log full request body as it might contain sensitive data
        body: req.body ? '[REDACTED]' : undefined,
      });
      
      next(error);
    });
  };
};

/**
 * 404 Not Found handler
 */
export const notFoundHandler = (req: Request, res: Response) => {
  logger.warn('Route not found', {
    path: req.originalUrl,
    method: req.method,
    ip: req.ip,
  });
  
  return errorResponse(
    res,
    {
      code: 'NOT_FOUND',
      message: `Cannot ${req.method} ${req.originalUrl}`,
    },
    404
  );
};
