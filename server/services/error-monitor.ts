import { logger } from '../config/logger';
import { setInterval, clearInterval } from 'timers';

interface ErrorContext {
  jobId?: string;
  userId?: string;
  fileType?: string;
  endpoint?: string;
  method?: string;
  statusCode?: number;
  [key: string]: unknown;
}

interface ErrorReport {
  timestamp: string;
  message: string;
  stack?: string;
  context: ErrorContext;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'analysis' | 'authentication' | 'database' | 'websocket' | 'python_bridge' | 'system';
}

class ErrorMonitor {
  private errorQueue: ErrorReport[] = [];
  private maxQueueSize = 1000;
  private flushInterval = 30000; // 30 seconds
  private flushTimer?: ReturnType<typeof setInterval>;

  constructor() {
    this.startFlushTimer();
  }

  // Capture error with context
  captureError(error: Error | string, context: ErrorContext = {}, severity: ErrorReport['severity'] = 'medium'): void {
    const errorReport: ErrorReport = {
      timestamp: new Date().toISOString(),
      message: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      context: this.sanitizeContext(context),
      severity,
      category: this.categorizeError(context)
    };

    // Add to queue
    this.errorQueue.push(errorReport);

    // Log immediately for critical errors
    if (severity === 'critical') {
      logger.error('[CRITICAL ERROR]', errorReport);
    }

    // Maintain queue size
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue = this.errorQueue.slice(-this.maxQueueSize);
    }
  }

  // Capture unhandled exceptions
  setupGlobalHandlers(): void {
    process.on('uncaughtException', (error: Error) => {
      this.captureError(error, {
        endpoint: 'global',
        method: 'uncaughtException'
      }, 'critical');
    });

    process.on('unhandledRejection', (reason: unknown, promise: Promise<unknown>) => {
      this.captureError(
        reason instanceof Error ? reason : new Error(String(reason)),
        {
          endpoint: 'global',
          method: 'unhandledRejection',
          promise: promise.toString?.() || 'unknown'
        },
        'high'
      );
    });
  }

  // Get error statistics
  getErrorStats(): {
    total: number;
    bySeverity: Record<string, number>;
    byCategory: Record<string, number>;
    recent: ErrorReport[];
  } {
    const bySeverity: Record<string, number> = {};
    const byCategory: Record<string, number> = {};

    this.errorQueue.forEach(error => {
      bySeverity[error.severity] = (bySeverity[error.severity] || 0) + 1;
      byCategory[error.category] = (byCategory[error.category] || 0) + 1;
    });

    return {
      total: this.errorQueue.length,
      bySeverity,
      byCategory,
      recent: this.errorQueue.slice(-10) // Last 10 errors
    };
  }

  // Flush error queue (would send to external service)
  private flush(): void {
    if (this.errorQueue.length === 0) return;

    const errorsToFlush = this.errorQueue.splice(0);
    
    // Log batch for now (in production, send to Sentry/DataDog)
    logger.info('[ERROR MONITOR] Flushing errors', {
      count: errorsToFlush.length,
      errors: errorsToFlush.map(e => ({
        timestamp: e.timestamp,
        message: e.message,
        severity: e.severity,
        category: e.category,
        context: e.context
      }))
    });
  }

  // Start flush timer
  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  // Categorize error based on context
  private categorizeError(context: ErrorContext): ErrorReport['category'] {
    if (context.endpoint?.includes('analysis') || context.endpoint?.includes('python')) {
      return 'analysis';
    }
    if (context.endpoint?.includes('auth') || context.endpoint?.includes('login')) {
      return 'authentication';
    }
    if (context.endpoint?.includes('db') || context.endpoint?.includes('database')) {
      return 'database';
    }
    if (context.endpoint?.includes('ws') || context.endpoint?.includes('websocket')) {
      return 'websocket';
    }
    if (context.endpoint?.includes('python') || context.endpoint?.includes('ml')) {
      return 'python_bridge';
    }
    return 'system';
  }

  // Sanitize context to remove sensitive data
  private sanitizeContext(context: ErrorContext): ErrorContext {
    const sanitized = { ...context };
    
    // Remove sensitive fields
    const sensitiveKeys = ['password', 'token', 'key', 'secret', 'authorization'];
    sensitiveKeys.forEach(key => {
      if (key in sanitized) {
        sanitized[key] = '[REDACTED]';
      }
    });

    return sanitized;
  }

  // Cleanup
  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flush(); // Final flush
  }
}

// Singleton instance
const errorMonitor = new ErrorMonitor();
export default errorMonitor;

// Convenience functions
export const captureError = (error: Error | string, context?: ErrorContext, severity?: ErrorReport['severity']) => {
  errorMonitor.captureError(error, context, severity);
};

export const setupErrorMonitoring = () => {
  errorMonitor.setupGlobalHandlers();
};
