import { AxiosRequestConfig } from 'axios';

export class ApiError extends Error {
  code: string;
  status: number;
  config?: AxiosRequestConfig;
  originalError?: any;

  constructor(
    message: string,
    code: string = 'API_ERROR',
    status: number = 0,
    config?: AxiosRequestConfig,
    originalError?: any
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.config = config;
    this.originalError = originalError;
    
    // Maintain proper stack trace in V8
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

export class NetworkError extends ApiError {
  constructor(message: string, config?: AxiosRequestConfig) {
    super(message, 'NETWORK_ERROR', 0, config);
    this.name = 'NetworkError';
  }
}

export class BadRequestError extends ApiError {
  data: any;

  constructor(message: string, data?: any, config?: AxiosRequestConfig) {
    super(message, 'BAD_REQUEST', 400, config);
    this.name = 'BadRequestError';
    this.data = data;
  }
}

export class UnauthorizedError extends ApiError {
  constructor(message: string = 'Unauthorized', config?: AxiosRequestConfig) {
    super(message, 'UNAUTHORIZED', 401, config);
    this.name = 'UnauthorizedError';
  }
}

export class ForbiddenError extends ApiError {
  constructor(message: string = 'Forbidden', config?: AxiosRequestConfig) {
    super(message, 'FORBIDDEN', 403, config);
    this.name = 'ForbiddenError';
  }
}

export class NotFoundError extends ApiError {
  constructor(message: string = 'Not Found', config?: AxiosRequestConfig) {
    super(message, 'NOT_FOUND', 404, config);
    this.name = 'NotFoundError';
  }
}

export class TimeoutError extends ApiError {
  constructor(message: string = 'Request Timeout', config?: AxiosRequestConfig) {
    super(message, 'TIMEOUT', 408, config);
    this.name = 'TimeoutError';
  }
}

export class RateLimitError extends ApiError {
  retryAfter?: number;

  constructor(
    message: string = 'Too Many Requests',
    config?: AxiosRequestConfig,
    retryAfter?: number
  ) {
    super(message, 'RATE_LIMIT', 429, config);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ServerError extends ApiError {
  data: any;

  constructor(message: string, data?: any, config?: AxiosRequestConfig) {
    super(message, 'SERVER_ERROR', 500, config);
    this.name = 'ServerError';
    this.data = data;
  }
}

export class ValidationError extends BadRequestError {
  fieldErrors: Record<string, string[]>;

  constructor(
    message: string = 'Validation Error',
    fieldErrors: Record<string, string[]> = {},
    data?: any,
    config?: AxiosRequestConfig
  ) {
    super(message, data || fieldErrors, config);
    this.name = 'ValidationError';
    this.fieldErrors = fieldErrors;
  }
}

export const isApiError = (error: any): error is ApiError => {
  return error instanceof ApiError || 
         (error && typeof error === 'object' && 
          'code' in error && 
          'status' in error);
};

export const isNetworkError = (error: any): error is NetworkError => {
  return error?.name === 'NetworkError' || 
         (isApiError(error) && error.code === 'NETWORK_ERROR');
};

export const isUnauthorizedError = (error: any): error is UnauthorizedError => {
  return error?.name === 'UnauthorizedError' || 
         (isApiError(error) && error.status === 401);
};

export const isForbiddenError = (error: any): error is ForbiddenError => {
  return error?.name === 'ForbiddenError' || 
         (isApiError(error) && error.status === 403);
};

export const isNotFoundError = (error: any): error is NotFoundError => {
  return error?.name === 'NotFoundError' || 
         (isApiError(error) && error.status === 404);
};

export const isServerError = (error: any): error is ServerError => {
  return error?.name === 'ServerError' || 
         (isApiError(error) && error.status >= 500);
};

export const isValidationError = (error: any): error is ValidationError => {
  return error?.name === 'ValidationError' || 
         (isApiError(error) && error.status === 400 && 'fieldErrors' in error);
};
