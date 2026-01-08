import { Response } from 'express';

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
    retryAfter?: number;
    [key: string]: any; // Allow additional properties
  };
  meta?: Record<string, any>;
  requestId?: string;
  timestamp?: string;
}

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly statusCode: number = 500,
    public readonly details?: any,
    public readonly retryAfter?: number
  ) {
    super(message);
    this.name = 'ApiError';
    
    // Maintain proper stack trace in V8
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
  }

  toJSON() {
    return {
      code: this.code,
      message: this.message,
      ...(this.details && { details: this.details }),
      ...(this.retryAfter && { retryAfter: this.retryAfter }),
    };
  }

  // Common error types as static methods
  static badRequest(message: string, details?: any) {
    return new ApiError('BAD_REQUEST', message, 400, details);
  }

  static unauthorized(message = 'Unauthorized') {
    return new ApiError('UNAUTHORIZED', message, 401);
  }

  static forbidden(message = 'Forbidden') {
    return new ApiError('FORBIDDEN', message, 403);
  }

  static notFound(resource: string) {
    return new ApiError('NOT_FOUND', `${resource} not found`, 404);
  }

  static conflict(message: string) {
    return new ApiError('CONFLICT', message, 409);
  }

  static tooManyRequests(message = 'Too many requests', retryAfter?: number) {
    return new ApiError('TOO_MANY_REQUESTS', message, 429, undefined, retryAfter);
  }

  static validationError(details: any) {
    return new ApiError('VALIDATION_ERROR', 'Validation failed', 422, details);
  }

  static internal(message = 'Internal server error') {
    return new ApiError('INTERNAL_SERVER_ERROR', message, 500);
  }
}

// Extend Express Response type
declare global {
  namespace Express {
    interface Response {
      requestId?: string;
    }
  }
}

export const successResponse = <T = any>(
  res: Response,
  data: T,
  meta?: Record<string, any>,
  statusCode = 200
): Response<ApiResponse<T>> => {
  const response: ApiResponse<T> = {
    success: true,
    data,
    timestamp: new Date().toISOString(),
  };

  if (meta && Object.keys(meta).length > 0) {
    response.meta = meta;
  }

  // Add request ID if available
  if (res.requestId) {
    response.requestId = res.requestId;
  }

  return res.status(statusCode).json(response);
};

export interface ErrorResponseOptions {
  code: string;
  message: string;
  details?: any;
  statusCode?: number;
  retryAfter?: number;
  [key: string]: any; // Allow additional properties
}

export const errorResponse = (
  res: Response,
  error: ErrorResponseOptions | Error | ApiError,
  statusCode = 400
): Response<ApiResponse> => {
  let response: ApiResponse;
  
  // Handle ApiError instances
  if (error instanceof ApiError) {
    response = {
      success: false,
      error: {
        code: error.code,
        message: error.message,
        ...(error.details && { details: error.details }),
        ...(error.retryAfter && { retryAfter: error.retryAfter }),
      },
      timestamp: new Date().toISOString(),
    };
    statusCode = error.statusCode;
  } 
  // Handle standard Error objects
  else if (error instanceof Error) {
    response = {
      success: false,
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message || 'An unexpected error occurred',
        ...(process.env.NODE_ENV !== 'production' && { stack: error.stack }),
      },
      timestamp: new Date().toISOString(),
    };
    statusCode = 500;
  }
  // Handle plain error objects
  else {
    const err = error as ErrorResponseOptions;
    response = {
      success: false,
      error: {
        code: err.code || 'UNKNOWN_ERROR',
        message: err.message || 'An unknown error occurred',
        ...(err.details && { details: err.details }),
        ...(err.retryAfter && { retryAfter: err.retryAfter }),
      },
      timestamp: new Date().toISOString(),
    };
    statusCode = err.statusCode || statusCode;
  }

  // Add request ID if available
  if (res.requestId) {
    response.requestId = res.requestId;
  }

  return res.status(statusCode).json(response);
};

export const serverError = (
  res: Response, 
  error: Error | string = 'Internal server error'
): Response<ApiResponse> => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorStack = typeof error === 'string' ? undefined : error.stack;
  
  // Log the error for debugging
  console.error('Server Error:', error);

  return errorResponse(
    res,
    {
      code: 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred',
      ...(process.env.NODE_ENV !== 'production' && { 
        details: errorMessage,
        stack: errorStack 
      }),
    },
    500
  );
};

export const notFoundResponse = (
  res: Response, 
  resource = 'Resource',
  details?: any
): Response<ApiResponse> => {
  return errorResponse(
    res,
    {
      code: 'NOT_FOUND',
      message: `${resource} not found`,
      ...(details && { details }),
    },
    404
  );
};

export const unauthorizedResponse = (
  res: Response, 
  message = 'Unauthorized',
  details?: any
): Response<ApiResponse> => {
  return errorResponse(
    res,
    {
      code: 'UNAUTHORIZED',
      message,
      ...(details && { details }),
    },
    401
  );
};

export const forbiddenResponse = (
  res: Response, 
  message = 'Forbidden',
  details?: any
): Response<ApiResponse> => {
  return errorResponse(
    res,
    {
      code: 'FORBIDDEN',
      message,
      ...(details && { details }),
    },
    403
  );
};

export const validationError = (
  res: Response, 
  details: any,
  message = 'Validation failed'
): Response<ApiResponse> => {
  return errorResponse(
    res,
    {
      code: 'VALIDATION_ERROR',
      message,
      details,
    },
    422
  );
};

// Export all error types for convenience
export const apiErrors = {
  badRequest: (message: string, details?: any) => 
    ApiError.badRequest(message, details),
  
  unauthorized: (message = 'Unauthorized') => 
    ApiError.unauthorized(message),
    
  forbidden: (message = 'Forbidden') => 
    ApiError.forbidden(message),
    
  notFound: (resource: string) => 
    ApiError.notFound(resource),
    
  conflict: (message: string) => 
    ApiError.conflict(message),
    
  tooManyRequests: (message = 'Too many requests', retryAfter?: number) => 
    ApiError.tooManyRequests(message, retryAfter),
    
  validation: (details: any) => 
    ApiError.validationError(details),
    
  internal: (message = 'Internal server error') => 
    ApiError.internal(message),
};
