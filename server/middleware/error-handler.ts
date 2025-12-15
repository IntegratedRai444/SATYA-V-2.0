import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

export class ApiError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public isOperational: boolean = true,
    public errors?: any[]
  ) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
  }
}

export class BadRequestError extends ApiError {
  constructor(message = 'Bad Request', errors?: any[]) {
    super(400, message, true, errors);
  }
}

export class UnauthorizedError extends ApiError {
  constructor(message = 'Unauthorized') {
    super(401, message, true);
  }
}

export class ForbiddenError extends ApiError {
  constructor(message = 'Forbidden') {
    super(403, message, true);
  }
}

export class NotFoundError extends ApiError {
  constructor(message = 'Not Found') {
    super(404, message, true);
  }
}

export class ConflictError extends ApiError {
  constructor(message = 'Conflict') {
    super(409, message, true);
  }
}

export class ValidationError extends ApiError {
  constructor(message = 'Validation Error', errors: any[] = []) {
    super(422, message, true, errors);
  }
}

export class RateLimitError extends ApiError {
  constructor(message = 'Too Many Requests') {
    super(429, message, true);
  }
}

export class InternalServerError extends ApiError {
  constructor(message = 'Internal Server Error') {
    super(500, message, false);
  }
}

export function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) {
  // Handle known error types
  if (err instanceof ApiError) {
    const { statusCode, message, errors, isOperational } = err;
    
    const response = {
      error: {
        code: statusCode,
        message,
        ...(errors && { errors }),
        ...(process.env.NODE_ENV !== 'production' && { stack: err.stack })
      },
      timestamp: new Date().toISOString(),
      path: req.path,
      method: req.method
    };

    // Log operational errors at warn level, others at error level
    if (isOperational) {
      logger.warn('Operational error', response);
    } else {
      logger.error('Unexpected error', { 
        ...response, 
        stack: err.stack 
      });
    }

    return res.status(statusCode).json(response);
  }

  // Handle JWT errors
  if (err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError') {
    const statusCode = 401;
    const response = {
      error: {
        code: statusCode,
        message: 'Invalid or expired token',
        ...(process.env.NODE_ENV !== 'production' && { details: err.message })
      },
      timestamp: new Date().toISOString()
    };
    
    logger.warn('Authentication error', { 
      ...response, 
      path: req.path, 
      method: req.method 
    });
    
    return res.status(statusCode).json(response);
  }

  // Handle validation errors (e.g., from Joi, Zod, etc.)
  if (err.name === 'ValidationError' || err.name === 'ZodError') {
    const statusCode = 422;
    const response = {
      error: {
        code: statusCode,
        message: 'Validation Error',
        ...(process.env.NODE_ENV !== 'production' && { details: err.message })
      },
      timestamp: new Date().toISOString()
    };
    
    logger.warn('Validation error', { 
      ...response, 
      path: req.path, 
      method: req.method 
    });
    
    return res.status(statusCode).json(response);
  }

  // Handle rate limiting errors
  if (err.name === 'RateLimitError') {
    const statusCode = 429;
    const response = {
      error: {
        code: statusCode,
        message: 'Too Many Requests',
        ...(process.env.NODE_ENV !== 'production' && { details: err.message })
      },
      timestamp: new Date().toISOString()
    };
    
    logger.warn('Rate limit exceeded', { 
      ...response, 
      path: req.path, 
      method: req.method,
      ip: req.ip
    });
    
    return res.status(statusCode).json(response);
  }

  // Handle database errors
  if (err.name === 'DatabaseError' || err.name.includes('Sequelize')) {
    const statusCode = 503;
    const response = {
      error: {
        code: statusCode,
        message: 'Service Unavailable',
        ...(process.env.NODE_ENV !== 'production' && { details: 'Database operation failed' })
      },
      timestamp: new Date().toISOString()
    };
    
    logger.error('Database error', { 
      error: err.message, 
      stack: err.stack, 
      path: req.path, 
      method: req.method 
    });
    
    return res.status(statusCode).json(response);
  }

  // Handle all other errors
  const statusCode = 500;
  const response = {
    error: {
      code: statusCode,
      message: 'Internal Server Error',
      ...(process.env.NODE_ENV !== 'production' && { details: err.message })
    },
    timestamp: new Date().toISOString()
  };
  
  logger.error('Unhandled error', { 
    error: err.message, 
    stack: err.stack, 
    path: req.path, 
    method: req.method,
    body: req.body,
    params: req.params,
    query: req.query,
    headers: req.headers
  });
  
  return res.status(statusCode).json(response);
}

export function notFoundHandler(req: Request, res: Response) {
  const statusCode = 404;
  const response = {
    error: {
      code: statusCode,
      message: 'Not Found',
      path: req.path,
      method: req.method
    },
    timestamp: new Date().toISOString()
  };
  
  logger.warn('Route not found', response);
  
  return res.status(statusCode).json(response);
}

export function asyncHandler(fn: Function) {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

// Global error handler for unhandled rejections
process.on('unhandledRejection', (reason: Error | any, promise: Promise<any>) => {
  logger.error('Unhandled Rejection at:', { 
    promise, 
    reason: reason?.message || reason,
    stack: reason?.stack 
  });
  
  // In production, you might want to gracefully shut down the server
  if (process.env.NODE_ENV === 'production') {
    // Consider using a process manager like PM2 to restart the process
    process.exit(1);
  }
});

// Global error handler for uncaught exceptions
process.on('uncaughtException', (error: Error) => {
  logger.error('Uncaught Exception:', { 
    error: error.message, 
    stack: error.stack 
  });
  
  // In production, you might want to gracefully shut down the server
  if (process.env.NODE_ENV === 'production') {
    // Consider using a process manager like PM2 to restart the process
    process.exit(1);
  }
});
