import { logger } from '../config/logger';

// Standardized error response format
export interface StandardError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    timestamp: string;
    requestId?: string;
  };
}

export interface ErrorDetails {
  code: string;
  message: string;
  originalError?: unknown;
  status?: number;
  statusText?: string;
  url?: string;
  timestamp: string;
  userId?: string;
  requestId?: string;
}

// Error code mappings for consistent error handling
export const ERROR_CODES = {
  // Authentication & Authorization
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',
  INVALID_TOKEN: 'INVALID_TOKEN',
  
  // Request Validation
  INVALID_REQUEST: 'INVALID_REQUEST',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  MISSING_REQUIRED_FIELD: 'MISSING_REQUIRED_FIELD',
  INVALID_FILE_TYPE: 'INVALID_FILE_TYPE',
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  
  // Service Errors
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  AI_ENGINE_UNAVAILABLE: 'AI_ENGINE_UNAVAILABLE',
  DATABASE_ERROR: 'DATABASE_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  
  // Analysis Errors
  ANALYSIS_ERROR: 'ANALYSIS_ERROR',
  ANALYSIS_TIMEOUT: 'ANALYSIS_TIMEOUT',
  MODEL_LOAD_FAILED: 'MODEL_LOAD_FAILED',
  INFERENCE_FAILED: 'INFERENCE_FAILED',
  
  // System Errors
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  CONCURRENT_LIMIT_EXCEEDED: 'CONCURRENT_LIMIT_EXCEEDED',
  
  // Python Service Specific
  PYTHON_DOWN: 'PYTHON_DOWN',
  PYTHON_AUTH_ERROR: 'PYTHON_AUTH_ERROR',
  PYTHON_FORBIDDEN: 'PYTHON_FORBIDDEN',
  PYTHON_NOT_FOUND: 'PYTHON_NOT_FOUND',
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];

// Standard error messages with timestamp
export const getErrorMessage = (code: ErrorCode): string => {
  const messages = {
    [ERROR_CODES.UNAUTHORIZED]: 'Authentication required',
    [ERROR_CODES.FORBIDDEN]: 'Access denied',
    [ERROR_CODES.TOKEN_EXPIRED]: 'Session expired, please login again',
    [ERROR_CODES.INVALID_TOKEN]: 'Invalid authentication token',
    
    [ERROR_CODES.INVALID_REQUEST]: 'Invalid request format',
    [ERROR_CODES.VALIDATION_ERROR]: 'Request validation failed',
    [ERROR_CODES.MISSING_REQUIRED_FIELD]: 'Required field is missing',
    [ERROR_CODES.INVALID_FILE_TYPE]: 'Unsupported file type',
    [ERROR_CODES.FILE_TOO_LARGE]: 'File size exceeds limit',
    
    [ERROR_CODES.SERVICE_UNAVAILABLE]: 'Service temporarily unavailable',
    [ERROR_CODES.AI_ENGINE_UNAVAILABLE]: 'AI engine temporarily unavailable',
    [ERROR_CODES.DATABASE_ERROR]: 'Database operation failed',
    [ERROR_CODES.NETWORK_ERROR]: 'Network connection failed',
    
    [ERROR_CODES.ANALYSIS_ERROR]: 'Analysis failed',
    [ERROR_CODES.ANALYSIS_TIMEOUT]: 'Analysis timed out',
    [ERROR_CODES.MODEL_LOAD_FAILED]: 'Failed to load ML model',
    [ERROR_CODES.INFERENCE_FAILED]: 'ML inference failed',
    
    [ERROR_CODES.INTERNAL_ERROR]: 'Internal server error',
    [ERROR_CODES.RATE_LIMIT_EXCEEDED]: 'Too many requests, please try again later',
    [ERROR_CODES.CONCURRENT_LIMIT_EXCEEDED]: 'Too many concurrent operations',
    
    [ERROR_CODES.PYTHON_DOWN]: 'AI service is down',
    [ERROR_CODES.PYTHON_AUTH_ERROR]: 'AI service authentication failed',
    [ERROR_CODES.PYTHON_FORBIDDEN]: 'AI service access forbidden',
    [ERROR_CODES.PYTHON_NOT_FOUND]: 'AI service endpoint not found',
  };
  
  return messages[code] || 'Unknown error occurred';
};

// Create standardized error response
export function createStandardError(
  code: ErrorCode,
  details?: Partial<ErrorDetails>,
  originalError?: unknown
): StandardError {
  const timestamp = new Date().toISOString();
  
  // Log the error with context
  logger.error('Standardized error created', {
    code,
    message: getErrorMessage(code),
    details,
    originalError: originalError instanceof Error ? originalError.message : originalError,
    timestamp,
  });

  return {
    success: false,
    error: {
      code,
      message: getErrorMessage(code),
      timestamp,
      details: {
        ...details,
        timestamp,
        originalError: originalError instanceof Error ? {
          name: originalError.name,
          message: originalError.message,
          stack: process.env.NODE_ENV === 'development' ? originalError.stack : undefined
        } : originalError
      }
    }
  };
}

// Format error for HTTP response
export function formatErrorResponse(
  code: ErrorCode,
  statusCode: number = 500,
  details?: Partial<ErrorDetails>,
  originalError?: unknown
): { statusCode: number; body: StandardError } {
  return {
    statusCode,
    body: createStandardError(code, details, originalError)
  };
}

// Convert various error types to standard format
export function normalizeError(error: unknown): StandardError {
  if (error && typeof error === 'object' && 'success' in error && error.success === false) {
    // Already a standardized error
    return error as StandardError;
  }

  if (error instanceof Error) {
    // Handle standard JavaScript errors
    return createStandardError(ERROR_CODES.INTERNAL_ERROR, {
      originalError: error.message,
      timestamp: new Date().toISOString()
    }, error);
  }

  // Handle unknown errors
  return createStandardError(ERROR_CODES.INTERNAL_ERROR, {
    originalError: String(error),
    timestamp: new Date().toISOString()
  }, error);
}

// Get HTTP status code for error type
export function getStatusCodeForError(code: ErrorCode): number {
  const statusMap: Record<ErrorCode, number> = {
    [ERROR_CODES.UNAUTHORIZED]: 401,
    [ERROR_CODES.FORBIDDEN]: 403,
    [ERROR_CODES.TOKEN_EXPIRED]: 401,
    [ERROR_CODES.INVALID_TOKEN]: 401,
    
    [ERROR_CODES.INVALID_REQUEST]: 400,
    [ERROR_CODES.VALIDATION_ERROR]: 422,
    [ERROR_CODES.MISSING_REQUIRED_FIELD]: 400,
    [ERROR_CODES.INVALID_FILE_TYPE]: 400,
    [ERROR_CODES.FILE_TOO_LARGE]: 413,
    
    [ERROR_CODES.SERVICE_UNAVAILABLE]: 503,
    [ERROR_CODES.AI_ENGINE_UNAVAILABLE]: 503,
    [ERROR_CODES.DATABASE_ERROR]: 500,
    [ERROR_CODES.NETWORK_ERROR]: 503,
    
    [ERROR_CODES.ANALYSIS_ERROR]: 500,
    [ERROR_CODES.ANALYSIS_TIMEOUT]: 504,
    [ERROR_CODES.MODEL_LOAD_FAILED]: 503,
    [ERROR_CODES.INFERENCE_FAILED]: 500,
    
    [ERROR_CODES.INTERNAL_ERROR]: 500,
    [ERROR_CODES.RATE_LIMIT_EXCEEDED]: 429,
    [ERROR_CODES.CONCURRENT_LIMIT_EXCEEDED]: 429,
    
    [ERROR_CODES.PYTHON_DOWN]: 503,
    [ERROR_CODES.PYTHON_AUTH_ERROR]: 502,
    [ERROR_CODES.PYTHON_FORBIDDEN]: 503,
    [ERROR_CODES.PYTHON_NOT_FOUND]: 404,
  };

  return statusMap[code] || 500;
}
