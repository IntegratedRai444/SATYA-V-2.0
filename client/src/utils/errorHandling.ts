import logger from './logger';

export type ErrorLevel = 'low' | 'medium' | 'high' | 'critical';

export interface ErrorContext {
  component?: string;
  action?: string;
  userId?: string;
  level?: ErrorLevel;
  [key: string]: any;
}

export interface ErrorWithContext extends Error {
  context?: ErrorContext;
}

/**
 * Classify error based on its type and context
 */
export const classifyError = (error: Error | ErrorWithContext): { type: string; severity: ErrorLevel } => {
  // Default values
  let type = 'unknown';
  let severity: ErrorLevel = 'medium';

  // Check for network errors
  if (
    error.message.includes('Network Error') ||
    error.message.includes('Failed to fetch') ||
    error.message.includes('timeout')
  ) {
    type = 'network';
    severity = 'high';
  }
  // Check for authentication errors
  else if (error.message.includes('401') || error.message.includes('unauthorized')) {
    type = 'auth';
    severity = 'high';
  }
  // Check for permission errors
  else if (error.message.includes('403') || error.message.includes('forbidden')) {
    type = 'permission';
    severity = 'high';
  }
  // Check for not found errors
  else if (error.message.includes('404') || error.message.includes('not found')) {
    type = 'not_found';
    severity = 'medium';
  }
  // Check for validation errors
  else if (error.name === 'ValidationError' || error.message.includes('validation')) {
    type = 'validation';
    severity = 'low';
  }
  // Check for chunk load errors (code splitting)
  else if (error.name === 'ChunkLoadError') {
    type = 'chunk_load';
    severity = 'critical';
  }

  // Override with context if provided
  if ('context' in error && error.context?.level) {
    severity = error.context.level;
  }

  return { type, severity };
};

/**
 * Handle error with appropriate logging and user feedback
 */
export const handleError = (
  error: Error | ErrorWithContext | unknown,
  options: {
    showToast?: boolean;
    logToConsole?: boolean;
    reportToService?: boolean;
    context?: ErrorContext;
  } = {}
): void => {
  const safeError = error instanceof Error ? error : new Error(String(error));
  const errorWithContext = safeError as ErrorWithContext;
  
  // Add context to error if provided
  if (options.context) {
    errorWithContext.context = {
      ...(errorWithContext.context || {}),
      ...options.context,
    };
  }

  // Classify the error
  const { type, severity } = classifyError(errorWithContext);

  // Log to console if enabled
  if (options.logToConsole !== false) {
    const logContext = {
      type,
      severity,
      ...(errorWithContext.context || {}),
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
      // TODO: Replace with your error tracking service integration
      // Example:
      // trackError(safeError, {
      //   level: severity,
      //   tags: { type },
      //   context: errorWithContext.context,
      // });
    } catch (e) {
      console.error('Failed to report error to service:', e);
    }
  }

  // Show user feedback if enabled
  if (options.showToast) {
    let message = 'An unexpected error occurred';
    
    // Customize message based on error type
    switch (type) {
      case 'network':
        message = 'Network error. Please check your connection and try again.';
        break;
      case 'auth':
        message = 'Session expired. Please log in again.';
        break;
      case 'chunk_load':
        message = 'A new version is available. Please refresh the page.';
        break;
      default:
        message = safeError.message || message;
    }

    // TODO: Replace with your toast implementation
    // toast.error(message);
    console.error('Toast:', message);
  }
};

/**
 * Log error with context
 */
export const logError = (
  error: Error | string,
  context: ErrorContext = {}
) => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorStack = typeof error === 'object' ? error.stack : undefined;
  
  // Log to console in development
  if (import.meta.env.DEV) {
    console.error('Error:', {
      message: errorMessage,
      stack: errorStack,
      context,
    });
  }

  // Log to error tracking service in production
  if (import.meta.env.PROD) {
    try {
      // Example of sending error to a custom error tracking endpoint
      // Replace this with your actual error tracking service integration
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

      // Example: Send to your error tracking API
      // fetch('/api/error-log', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(errorData),
      // });

      console.error('Error reported to tracking service:', errorData);
    } catch (trackingError) {
      console.error('Failed to send error to tracking service:', trackingError);
    }
  }
};

// Example usage:
// try {
//   // Your code here
// } catch (error) {
//   logError(error, {
//     component: 'ComponentName',
//     action: 'fetchData',
//     userId: 'user123',
//     additionalInfo: 'More context here'
//   });
// }
