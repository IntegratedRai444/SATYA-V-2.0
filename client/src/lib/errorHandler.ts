/**
 * Centralized Error Handling System
 * Provides consistent error handling, classification, and user feedback
 */

import logger from './logger';
import type { ApiError } from '../types/api';

// ============================================================================
// Error Types
// ============================================================================

export type ErrorType =
  | 'network'
  | 'authentication'
  | 'authorization'
  | 'validation'
  | 'not_found'
  | 'server'
  | 'timeout'
  | 'rate_limit'
  | 'unknown';

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// ============================================================================
// Error Handler Options
// ============================================================================

export interface ErrorHandlerOptions {
  showToast?: boolean;
  logToConsole?: boolean;
  reportToService?: boolean;
  fallbackMessage?: string;
  retry?: boolean;
  maxRetries?: number;
  onRetry?: () => void;
}

// ============================================================================
// Classified Error Interface
// ============================================================================

export interface ClassifiedError {
  type: ErrorType;
  severity: ErrorSeverity;
  message: string;
  userMessage: string;
  code?: string;
  statusCode?: number;
  details?: Record<string, unknown>;
  stack?: string;
  timestamp: string;
  recoverable: boolean;
  retryable: boolean;
}

// ============================================================================
// Error Messages
// ============================================================================

const ERROR_MESSAGES: Record<ErrorType, { title: string; message: string }> = {
  network: {
    title: 'Connection Error',
    message: 'Unable to connect to the server. Please check your internet connection and try again.',
  },
  authentication: {
    title: 'Authentication Required',
    message: 'Your session has expired. Please log in again to continue.',
  },
  authorization: {
    title: 'Access Denied',
    message: 'You do not have permission to perform this action.',
  },
  validation: {
    title: 'Invalid Input',
    message: 'Please check your input and try again.',
  },
  not_found: {
    title: 'Not Found',
    message: 'The requested resource could not be found.',
  },
  server: {
    title: 'Server Error',
    message: 'An error occurred on the server. Please try again later.',
  },
  timeout: {
    title: 'Request Timeout',
    message: 'The request took too long to complete. Please try again.',
  },
  rate_limit: {
    title: 'Too Many Requests',
    message: 'You have made too many requests. Please wait a moment and try again.',
  },
  unknown: {
    title: 'Unexpected Error',
    message: 'An unexpected error occurred. Please try again.',
  },
};

// ============================================================================
// Error Handler Class
// ============================================================================

class ErrorHandler {
  private retryCount = new Map<string, number>();
  private readonly MAX_RETRY_ATTEMPTS = 3;
  private readonly RETRY_DELAY = 1000; // 1 second

  /**
   * Classify an error based on its properties
   */
  classify(error: unknown): ClassifiedError {
    const timestamp = new Date().toISOString();

    // Handle API errors
    if (this.isApiError(error)) {
      return this.classifyApiError(error, timestamp);
    }

    // Handle Axios errors
    if (this.isAxiosError(error)) {
      return this.classifyAxiosError(error, timestamp);
    }

    // Handle standard Error objects
    if (error instanceof Error) {
      return this.classifyStandardError(error, timestamp);
    }

    // Handle unknown errors
    return {
      type: 'unknown',
      severity: 'medium',
      message: String(error),
      userMessage: ERROR_MESSAGES.unknown.message,
      timestamp,
      recoverable: false,
      retryable: false,
    };
  }

  /**
   * Classify API error
   */
  private classifyApiError(error: ApiError, timestamp: string): ClassifiedError {
    const statusCode = error.statusCode;
    let type: ErrorType = 'unknown';
    let severity: ErrorSeverity = 'medium';
    let recoverable = false;
    let retryable = false;

    if (statusCode === 401) {
      type = 'authentication';
      severity = 'high';
      recoverable = true;
    } else if (statusCode === 403) {
      type = 'authorization';
      severity = 'high';
    } else if (statusCode === 404) {
      type = 'not_found';
      severity = 'low';
    } else if (statusCode === 422 || statusCode === 400) {
      type = 'validation';
      severity = 'low';
      recoverable = true;
    } else if (statusCode === 429) {
      type = 'rate_limit';
      severity = 'medium';
      retryable = true;
    } else if (statusCode === 408 || statusCode === 504) {
      type = 'timeout';
      severity = 'medium';
      retryable = true;
    } else if (statusCode >= 500) {
      type = 'server';
      severity = 'high';
      retryable = true;
    }

    return {
      type,
      severity,
      message: error.message,
      userMessage: ERROR_MESSAGES[type].message,
      code: error.code,
      statusCode: error.statusCode,
      details: error.details,
      timestamp,
      recoverable,
      retryable,
    };
  }

  /**
   * Classify Axios error
   */
  private classifyAxiosError(error: any, timestamp: string): ClassifiedError {
    if (error.response) {
      // Server responded with error status
      return this.classifyApiError(
        {
          code: error.code || 'AXIOS_ERROR',
          message: error.message,
          statusCode: error.response.status,
          details: error.response.data,
          timestamp,
        },
        timestamp
      );
    } else if (error.request) {
      // Request made but no response
      return {
        type: 'network',
        severity: 'high',
        message: error.message,
        userMessage: ERROR_MESSAGES.network.message,
        code: error.code,
        timestamp,
        recoverable: true,
        retryable: true,
      };
    } else {
      // Error setting up request
      return {
        type: 'unknown',
        severity: 'medium',
        message: error.message,
        userMessage: ERROR_MESSAGES.unknown.message,
        timestamp,
        recoverable: false,
        retryable: false,
      };
    }
  }

  /**
   * Classify standard Error
   */
  private classifyStandardError(error: Error, timestamp: string): ClassifiedError {
    let type: ErrorType = 'unknown';
    let retryable = false;

    // Check error message for common patterns
    if (error.message.includes('network') || error.message.includes('fetch')) {
      type = 'network';
      retryable = true;
    } else if (error.message.includes('timeout')) {
      type = 'timeout';
      retryable = true;
    } else if (error.message.includes('auth')) {
      type = 'authentication';
    }

    return {
      type,
      severity: 'medium',
      message: error.message,
      userMessage: ERROR_MESSAGES[type].message,
      stack: error.stack,
      timestamp,
      recoverable: false,
      retryable,
    };
  }

  /**
   * Handle an error with specified options
   */
  async handle(error: unknown, options: ErrorHandlerOptions = {}): Promise<void> {
    const classified = this.classify(error);

    // Log error
    if (options.logToConsole !== false) {
      logger.error(
        `${classified.type} error: ${classified.message}`,
        error instanceof Error ? error : undefined,
        {
          type: classified.type,
          severity: classified.severity,
          statusCode: classified.statusCode,
          code: classified.code,
        }
      );
    }

    // Report to service
    if (options.reportToService && classified.severity === 'high' || classified.severity === 'critical') {
      await this.reportError(classified);
    }

    // Show toast notification
    if (options.showToast !== false) {
      this.showErrorToast(classified, options.fallbackMessage);
    }

    // Handle retry logic
    if (options.retry && classified.retryable) {
      await this.handleRetry(error, options);
    }
  }

  /**
   * Handle retry logic
   */
  private async handleRetry(error: unknown, options: ErrorHandlerOptions): Promise<void> {
    const errorKey = this.getErrorKey(error);
    const currentRetries = this.retryCount.get(errorKey) || 0;
    const maxRetries = options.maxRetries || this.MAX_RETRY_ATTEMPTS;

    if (currentRetries < maxRetries) {
      this.retryCount.set(errorKey, currentRetries + 1);
      
      logger.info(`Retrying operation (attempt ${currentRetries + 1}/${maxRetries})`);
      
      // Wait before retry with exponential backoff
      const delay = this.RETRY_DELAY * Math.pow(2, currentRetries);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      if (options.onRetry) {
        options.onRetry();
      }
    } else {
      this.retryCount.delete(errorKey);
      logger.warn('Max retry attempts reached');
    }
  }

  /**
   * Get unique key for error (for retry tracking)
   */
  private getErrorKey(error: unknown): string {
    if (error instanceof Error) {
      return `${error.name}:${error.message}`;
    }
    return String(error);
  }

  /**
   * Show error toast notification
   */
  private showErrorToast(error: ClassifiedError, fallbackMessage?: string): void {
    // This will be implemented with the toast system
    // For now, just log
    logger.debug('Would show toast', {
      title: ERROR_MESSAGES[error.type].title,
      message: fallbackMessage || error.userMessage,
      type: error.severity,
    });
  }

  /**
   * Report error to remote service
   */
  private async reportError(error: ClassifiedError): Promise<void> {
    try {
      // This would integrate with error tracking service (Sentry, LogRocket, etc.)
      logger.debug('Would report error to service', { error });
    } catch (reportError) {
      logger.warn('Failed to report error', reportError as Error);
    }
  }

  /**
   * Get user-friendly error message
   */
  getUserMessage(error: unknown): string {
    const classified = this.classify(error);
    return classified.userMessage;
  }

  /**
   * Check if error is a network error
   */
  isNetworkError(error: unknown): boolean {
    const classified = this.classify(error);
    return classified.type === 'network';
  }

  /**
   * Check if error is an authentication error
   */
  isAuthError(error: unknown): boolean {
    const classified = this.classify(error);
    return classified.type === 'authentication';
  }

  /**
   * Check if error should be retried
   */
  shouldRetry(error: unknown): boolean {
    const classified = this.classify(error);
    return classified.retryable;
  }

  /**
   * Check if error is recoverable
   */
  isRecoverable(error: unknown): boolean {
    const classified = this.classify(error);
    return classified.recoverable;
  }

  /**
   * Type guard for API errors
   */
  private isApiError(error: unknown): error is ApiError {
    return (
      typeof error === 'object' &&
      error !== null &&
      'code' in error &&
      'message' in error &&
      'statusCode' in error
    );
  }

  /**
   * Type guard for Axios errors
   */
  private isAxiosError(error: unknown): boolean {
    return (
      typeof error === 'object' &&
      error !== null &&
      'isAxiosError' in error &&
      (error as any).isAxiosError === true
    );
  }

  /**
   * Clear retry count for specific error
   */
  clearRetryCount(error: unknown): void {
    const errorKey = this.getErrorKey(error);
    this.retryCount.delete(errorKey);
  }

  /**
   * Clear all retry counts
   */
  clearAllRetryCounts(): void {
    this.retryCount.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

const errorHandler = new ErrorHandler();

// ============================================================================
// Exported Functions
// ============================================================================

export function handleError(error: unknown, options?: ErrorHandlerOptions): Promise<void> {
  return errorHandler.handle(error, options);
}

export function classifyError(error: unknown): ClassifiedError {
  return errorHandler.classify(error);
}

export function getUserMessage(error: unknown): string {
  return errorHandler.getUserMessage(error);
}

export function isNetworkError(error: unknown): boolean {
  return errorHandler.isNetworkError(error);
}

export function isAuthError(error: unknown): boolean {
  return errorHandler.isAuthError(error);
}

export function shouldRetry(error: unknown): boolean {
  return errorHandler.shouldRetry(error);
}

export function isRecoverable(error: unknown): boolean {
  return errorHandler.isRecoverable(error);
}

export default errorHandler;
export { ErrorHandler };
