import { Response } from 'express';

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
    retryAfter?: number;
    [key: string]: unknown; // Allow additional properties
  };
  meta?: Record<string, unknown>;
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
    public readonly details?: unknown,
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
    const result: Record<string, unknown> = {
      code: this.code,
      message: this.message,
    };
    
    if (this.details) {
      result.details = this.details;
    }
    
    if (this.retryAfter !== undefined) {
      result.retryAfter = this.retryAfter;
    }
    
    return result;
  }

  // Common error types as static methods
  static badRequest(message: string, details?: unknown) {
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

  static validationError(details: unknown) {
    return new ApiError('VALIDATION_ERROR', 'Validation failed', 422, details);
  }

  static internal(message = 'Internal server error') {
    return new ApiError('INTERNAL_SERVER_ERROR', message, 500);
  }
}

// Extend Express Response type
declare module 'express' {
  interface Response {
    requestId?: string;
  }
}

export const successResponse = <T = unknown>(
  res: Response,
  data: T,
  meta?: Record<string, unknown>,
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
  details?: unknown;
  statusCode?: number;
  retryAfter?: number;
  [key: string]: unknown; // Allow additional properties
}

interface ErrorObject {
  code: string;
  message: string;
  details?: unknown;
  retryAfter?: number;
  stack?: string;
  [key: string]: unknown;
}

export const errorResponse = (
  res: Response,
  error: ErrorResponseOptions | Error | ApiError,
  statusCode = 400
): Response<ApiResponse> => {
  let response: ApiResponse;
  
  // Handle ApiError instances
  if (error instanceof ApiError) {
    const errorObj: ErrorObject = {
      code: error.code,
      message: error.message,
    };
    
    if (error.details) {
      errorObj.details = error.details;
    }
    
    if (error.retryAfter !== undefined) {
      errorObj.retryAfter = error.retryAfter;
    }
    
    response = {
      success: false,
      error: errorObj,
      timestamp: new Date().toISOString(),
    };
    statusCode = error.statusCode;
  } 
  // Handle standard Error objects
  else if (error instanceof Error) {
    const errorObj: ErrorObject = {
      code: 'INTERNAL_ERROR',
      message: error.message || 'An unexpected error occurred',
    };
    
    if (process.env.NODE_ENV !== 'production' && error.stack) {
      errorObj.stack = error.stack;
    }
    
    response = {
      success: false,
      error: errorObj,
      timestamp: new Date().toISOString(),
    };
    statusCode = 500;
  }
  // Handle plain error objects
  else {
    const err = error as ErrorResponseOptions;
    const errorObj: ErrorObject = {
      code: err.code || 'UNKNOWN_ERROR',
      message: err.message || 'An unknown error occurred',
    };
    
    if (err.details) {
      errorObj.details = err.details;
    }
    
    if (err.retryAfter !== undefined) {
      errorObj.retryAfter = err.retryAfter;
    }
    
    response = {
      success: false,
      error: errorObj,
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
  message: string,
  statusCode: number = 500
): Response<ApiResponse> => {
  const response: ApiResponse = {
    success: false,
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: message
    },
    timestamp: new Date().toISOString()
  };
  
  if (res.requestId) {
    response.requestId = res.requestId;
  }

  return res.status(statusCode).json(response);
};

export const notFoundResponse = (
  res: Response, 
  resource = 'Resource',
  details?: unknown
): Response<ApiResponse> => {
  const errorDetails: Record<string, unknown> = {
    code: 'NOT_FOUND',
    message: `${resource} not found`,
  };
  
  if (details) {
    errorDetails.details = details;
  }
  
  return errorResponse(res, errorDetails as ErrorResponseOptions, 404);
};

export const unauthorizedResponse = (
  res: Response, 
  message = 'Unauthorized',
  details?: unknown
): Response<ApiResponse> => {
  const errorDetails: Record<string, unknown> = {
    code: 'UNAUTHORIZED',
    message,
  };
  
  if (details) {
    errorDetails.details = details;
  }
  
  return errorResponse(res, errorDetails as ErrorResponseOptions, 401);
};

export const forbiddenResponse = (
  res: Response, 
  message = 'Forbidden',
  details?: unknown
): Response<ApiResponse> => {
  const errorDetails: Record<string, unknown> = {
    code: 'FORBIDDEN',
    message,
  };
  
  if (details) {
    errorDetails.details = details;
  }
  
  return errorResponse(res, errorDetails as ErrorResponseOptions, 403);
};

export const validationError = (
  res: Response, 
  details: unknown,
  message = 'Validation failed'
): Response<ApiResponse> => {
  const errorDetails: Record<string, unknown> = {
    code: 'VALIDATION_ERROR',
    message,
    details,
  };
  
  return errorResponse(res, errorDetails as ErrorResponseOptions, 422);
};

// Export all error types for convenience
export const apiErrors = {
  badRequest: (message: string, details?: unknown) => 
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
    
  validation: (details: unknown) => 
    ApiError.validationError(details),
    
  internal: (message = 'Internal server error') => 
    ApiError.internal(message),
};
