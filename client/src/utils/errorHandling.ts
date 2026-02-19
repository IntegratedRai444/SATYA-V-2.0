import logger from './logger';
import { apiClient } from '../lib/api/client';

declare const process: {
  env: {
    NODE_ENV?: string;
  };
};

export type ErrorLevel = 'low' | 'medium' | 'high' | 'critical';

import { toast } from '@/components/ui/use-toast';

export interface ErrorContext {
  component?: string;
  action?: string;
  userId?: string;
  additionalInfo?: Record<string, unknown>;
}

/**
 * Classify error for better handling
 */
export class ClassifiedError extends Error {
  constructor(
    message: string,
    public type: 'network' | 'auth' | 'chunk_load' | 'unknown' = 'unknown',
    public severity: 'low' | 'medium' | 'high' = 'medium',
    public userMessage?: string
  ) {
    super(message);
    this.type = type;
    this.severity = severity;
    this.userMessage = userMessage;
  }
}

// Error message constants
const ERROR_MESSAGES = {
  network: {
    title: 'Network Error',
    message: 'Network error. Please check your connection and try again.'
  },
  auth: {
    title: 'Authentication Error',
    message: 'Session expired. Please log in again.'
  },
  chunk_load: {
    title: 'Update Required',
    message: 'A new version is available. Please refresh the page.'
  },
  unknown: {
    title: 'Error',
    message: 'An unexpected error occurred.'
  }
} as const;

/**
 * Classify error type and severity
 */
export const classifyError = (error: ClassifiedError | Error): { type: keyof typeof ERROR_MESSAGES; severity: string } => {
  if (error instanceof ClassifiedError) {
    return {
      type: error.type as keyof typeof ERROR_MESSAGES,
      severity: error.severity,
    };
  }

  // Default classification for unclassified errors
  return {
    type: 'unknown',
    severity: 'medium',
  };
};

/**
 * Handle errors with user feedback
 */
export const handleError = async (
  error: Error | ClassifiedError | unknown,
  options: {
    showToast?: boolean;
    logToConsole?: boolean;
    reportToService?: boolean;
    fallbackMessage?: string;
    context?: ErrorContext;
  } = {}
): Promise<void> => {
  const safeError = error instanceof Error ? error : new Error(String(error));
  let message = safeError.message || options.fallbackMessage || 'An unexpected error occurred';

  // Add context to error if provided
  const errorWithContext = options.context
    ? {
        ...(options.context || {}),
        ...options.context,
      }
    : {};

  // Classify the error
  const { type, severity } = classifyError(safeError);

  // Customize message based on error type
  if (ERROR_MESSAGES[type]) {
    message = ERROR_MESSAGES[type].message;
  }

  // Log to console if enabled
  if (options.logToConsole !== false) {
    const logContext = {
      type,
      severity,
      ...errorWithContext,
    };

    if (severity === 'high' || severity === 'critical') {
      logger.error(safeError.message, safeError, logContext);
    } else {
      logger.warn(safeError.message, logContext);
    }
  }

  // Report to error tracking service if enabled
  if (options.reportToService && process.env.NODE_ENV === 'production') {
    try {
      // Send to error tracking API
      const errorData = {
        timestamp: new Date().toISOString(),
        message: safeError.message,
        stack: safeError.stack,
        context: {
          ...errorWithContext,
          // Add any additional context you want to track
          url: window.location.href,
          userAgent: navigator.userAgent,
        },
      };

      // Send to your error tracking API
      await apiClient.post('/error-log', errorData);

    } catch (trackingError) {
      // Failed to send error to tracking service
      logger.error('Failed to send error to tracking service', trackingError as Error);
    }
  }

  // Show user feedback if enabled
  if (options.showToast !== false) {
    toast({
      title: ERROR_MESSAGES[type]?.title || 'Error',
      description: message,
      variant: 'destructive',
    });
  }
};

/**
 * Log error with context
 */
export const logError = async (
  error: Error | string,
  context: ErrorContext = {}
) => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorStack = typeof error === 'object' ? error.stack : undefined;

  // Log to error tracking service in production
  if (import.meta.env.PROD) {
    try {
      // Send to error tracking API
      const errorData = {
        timestamp: new Date().toISOString(),
        message: errorMessage,
        stack: errorStack,
        context: {
          ...context,
          // Add any additional context you want to track
          url: window.location.href,
          userAgent: navigator.userAgent,
        },
      };

      // Send to your error tracking API
      await apiClient.post('/error-log', errorData);

          } catch {
      // Failed to send error to tracking service
    }
  }
};

