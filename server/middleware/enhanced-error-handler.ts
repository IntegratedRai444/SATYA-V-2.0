import { Request, Response, NextFunction } from 'express';
import { logger } from '../config';
import { 
  createErrorResponse, 
  ErrorCategory, 
  ERROR_CODES, 
  ERROR_MESSAGES 
} from '../types/api-responses';

export interface EnhancedError extends Error {
  statusCode?: number;
  code?: string;
  category?: ErrorCategory;
  userMessage?: string;
  suggestions?: string[];
  retryable?: boolean;
  retryAfter?: number;
}

/**
 * Enhanced error handler middleware with user-friendly messages
 */
export function enhancedErrorHandler(
  error: EnhancedError,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Log the error with context
  logger.error('Request error occurred', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    userAgent: req.get('User-Agent'),
    ip: req.ip,
    userId: (req as any).user?.userId
  });

  // Determine error category and response
  const errorResponse = categorizeError(error);
  
  // Set appropriate status code
  const statusCode = error.statusCode || errorResponse.statusCode || 500;
  
  res.status(statusCode).json(
    createErrorResponse(
      errorResponse.code,
      errorResponse.message,
      errorResponse.details,
      errorResponse.suggestions
    )
  );
}

/**
 * Categorize errors and provide appropriate responses
 */
function categorizeError(error: EnhancedError): {
  code: string;
  message: string;
  details?: string;
  suggestions: string[];
  statusCode: number;
} {
  // Handle known error codes
  if (error.code && ERROR_MESSAGES[error.code as keyof typeof ERROR_MESSAGES]) {
    const errorInfo = ERROR_MESSAGES[error.code as keyof typeof ERROR_MESSAGES];
    return {
      code: error.code,
      message: errorInfo.message,
      details: error.message,
      suggestions: errorInfo.suggestions,
      statusCode: getStatusCodeForError(error.code)
    };
  }

  // Handle file upload errors
  if (error.message.includes('File too large')) {
    return {
      code: ERROR_CODES.FILE_TOO_LARGE,
      message: ERROR_MESSAGES[ERROR_CODES.FILE_TOO_LARGE].message,
      details: error.message,
      suggestions: ERROR_MESSAGES[ERROR_CODES.FILE_TOO_LARGE].suggestions,
      statusCode: 413
    };
  }

  if (error.message.includes('Unexpected field') || error.message.includes('Unexpected file')) {
    return {
      code: ERROR_CODES.INVALID_FILE_TYPE,
      message: ERROR_MESSAGES[ERROR_CODES.INVALID_FILE_TYPE].message,
      details: error.message,
      suggestions: ERROR_MESSAGES[ERROR_CODES.INVALID_FILE_TYPE].suggestions,
      statusCode: 400
    };
  }

  // Handle authentication errors
  if (error.message.includes('token') || error.message.includes('auth')) {
    return {
      code: ERROR_CODES.INVALID_TOKEN,
      message: 'Authentication failed',
      details: error.message,
      suggestions: ['Please log in again', 'Check your authentication token', 'Clear browser cache and cookies'],
      statusCode: 401
    };
  }

