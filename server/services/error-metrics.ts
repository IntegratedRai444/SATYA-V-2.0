import { logger } from '../config/logger';
import { setInterval, clearInterval } from 'timers';

export interface ErrorMetric {
  timestamp: string;
  code: string;
  message: string;
  endpoint?: string;
  method?: string;
  statusCode?: number;
  userId?: string;
  requestId?: string;
  userAgent?: string;
  ip?: string;
  stack?: string;
  recoveryTime?: number;
}

export interface ErrorMetricsSummary {
  totalErrors: number;
  errorsByCode: Record<string, number>;
  errorsByEndpoint: Record<string, number>;
  errorsByHour: Record<string, number>;
  recentErrors: ErrorMetric[];
  topErrors: Array<{
    code: string;
    count: number;
    message: string;
  }>;
}

class ErrorMetricsService {
  private errors: ErrorMetric[] = [];
  private maxErrors: number = 1000; // Keep last 1000 errors
  private metricsInterval: ReturnType<typeof setInterval> | null = null;
  private readonly METRICS_INTERVAL = 60000; // 1 minute

  constructor() {
    // Start periodic metrics processing
    this.startMetricsProcessing();
    logger.info('ErrorMetricsService initialized');
  }

  // Record an error metric
  recordError(error: ErrorMetric): void {
    const metric: ErrorMetric = {
      ...error,
      timestamp: error.timestamp || new Date().toISOString()
    };

    this.errors.push(metric);

    // Keep only recent errors
    if (this.errors.length > this.maxErrors) {
      this.errors = this.errors.slice(-this.maxErrors);
    }

    // Log critical errors immediately
    if (error.statusCode && error.statusCode >= 500) {
      logger.error('Critical error recorded', metric);
    }
  }

  // Extract error information from various sources
  extractErrorFromRequest(
    error: Error,
    req?: {
      path?: string;
      url?: string;
      method?: string;
      user?: { id?: string };
      headers?: Record<string, string>;
      ip?: string;
      connection?: { remoteAddress?: string };
    },
    additionalContext?: Record<string, unknown>
  ): ErrorMetric {
    const metric: ErrorMetric = {
      timestamp: new Date().toISOString(),
      code: (error as Error & { code?: string }).code || 'UNKNOWN_ERROR',
      message: error.message,
      stack: error.stack,
      ...additionalContext
    };

    // Extract request information if available
    if (req) {
      metric.endpoint = req.path || req.url;
      metric.method = req.method;
      metric.userId = req.user?.id;
      metric.requestId = req.headers?.['x-request-id'];
      metric.userAgent = req.headers?.['user-agent'];
      metric.ip = req.ip || req.connection?.remoteAddress;
    }

    // Extract status code if available
    if ((error as Error & { status?: number }).status) {
      metric.statusCode = (error as Error & { status?: number }).status;
    } else if ((error as Error & { response?: { status?: number } }).response?.status) {
      metric.statusCode = (error as Error & { response?: { status?: number } }).response!.status;
    }

    return metric;
  }

  // Get error metrics summary
  getMetricsSummary(): ErrorMetricsSummary {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

    // Filter errors from last hour
    const recentErrors = this.errors.filter(e => 
      new Date(e.timestamp) > oneHourAgo
    );

    // Group errors by code
    const errorsByCode: Record<string, number> = {};
    recentErrors.forEach(error => {
      errorsByCode[error.code] = (errorsByCode[error.code] || 0) + 1;
    });

    // Group errors by endpoint
    const errorsByEndpoint: Record<string, number> = {};
    recentErrors.forEach(error => {
      if (error.endpoint) {
        errorsByEndpoint[error.endpoint] = (errorsByEndpoint[error.endpoint] || 0) + 1;
      }
    });

    // Group errors by hour
    const errorsByHour: Record<string, number> = {};
    recentErrors.forEach(error => {
      const hour = new Date(error.timestamp).toISOString().slice(0, 13); // YYYY-MM-DDTHH
      errorsByHour[hour] = (errorsByHour[hour] || 0) + 1;
    });

    // Get top errors
    const topErrors = Object.entries(errorsByCode)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([code, count]) => ({
        code,
        count,
        message: recentErrors.find(e => e.code === code)?.message || 'Unknown error'
      }));

    return {
      totalErrors: recentErrors.length,
      errorsByCode,
      errorsByEndpoint,
      errorsByHour,
      recentErrors: recentErrors.slice(-50), // Last 50 errors
      topErrors
    };
  }

  // Get error rate for a specific time period
  getErrorRate(minutes: number = 60): number {
    const cutoff = new Date(Date.now() - minutes * 60 * 1000);
    const errorsInPeriod = this.errors.filter(e => 
      new Date(e.timestamp) > cutoff
    ).length;
    
    return errorsInPeriod / minutes; // errors per minute
  }

  // Get critical errors (5xx status codes)
  getCriticalErrors(limit: number = 50): ErrorMetric[] {
    return this.errors
      .filter(e => e.statusCode && e.statusCode >= 500)
      .slice(-limit)
      .reverse();
  }

  // Check if error rate is above threshold
  isErrorRateAboveThreshold(threshold: number = 10, windowMinutes: number = 5): boolean {
    const errorRate = this.getErrorRate(windowMinutes);
    return errorRate > threshold;
  }

  // Get recovery metrics
  getRecoveryMetrics(): {
    averageRecoveryTime: number;
    recoveryRate: number;
  } {
    const errorsWithRecovery = this.errors.filter(e => e.recoveryTime);
    
    if (errorsWithRecovery.length === 0) {
      return { averageRecoveryTime: 0, recoveryRate: 0 };
    }

    const averageRecoveryTime = errorsWithRecovery.reduce(
      (sum, e) => sum + (e.recoveryTime || 0), 0
    ) / errorsWithRecovery.length;

    const recoveryRate = errorsWithRecovery.length / this.errors.length;

    return { averageRecoveryTime, recoveryRate };
  }

  // Start periodic metrics processing
  private startMetricsProcessing(): void {
    this.metricsInterval = setInterval(() => {
      this.processMetrics();
    }, this.METRICS_INTERVAL);
  }

  // Process metrics periodically
  private processMetrics(): void {
    const summary = this.getMetricsSummary();
    
    // Log high error rates
    if (this.isErrorRateAboveThreshold()) {
      logger.warn('High error rate detected', {
        errorRate: this.getErrorRate(),
        totalErrors: summary.totalErrors,
        topErrors: summary.topErrors.slice(0, 3)
      });
    }

    // Log critical errors
    const criticalErrors = this.getCriticalErrors(5);
    if (criticalErrors.length > 0) {
      logger.error('Critical errors detected', criticalErrors);
    }
  }

  // Cleanup old errors
  cleanup(): void {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const beforeCount = this.errors.length;
    
    this.errors = this.errors.filter(e => new Date(e.timestamp) > oneDayAgo);
    
    const cleaned = beforeCount - this.errors.length;
    if (cleaned > 0) {
      logger.info(`Cleaned up ${cleaned} old error metrics`);
    }
  }

  // Stop metrics processing
  stop(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }
    logger.info('ErrorMetricsService stopped');
  }

  // Reset all metrics
  reset(): void {
    this.errors = [];
    logger.info('ErrorMetricsService reset');
  }
}

// Export singleton instance
export const errorMetrics = new ErrorMetricsService();

// Export middleware for automatic error tracking
export function errorMetricsMiddleware() {
  return (error: Error, req: Record<string, unknown>, res: unknown, next: (err?: unknown) => void) => {
    // Record the error
    const metric = errorMetrics.extractErrorFromRequest(error, req as {
      path?: string;
      url?: string;
      method?: string;
      user?: { id?: string };
      headers?: Record<string, string>;
      ip?: string;
      connection?: { remoteAddress?: string };
      startTime?: number;
    }, {
      recoveryTime: Date.now() - ((req as { startTime?: number }).startTime || Date.now())
    });
    
    errorMetrics.recordError(metric);
    
    // Continue with error handling
    next(error);
  };
}

// Export utility function for manual error recording
export function recordError(
  error: Error,
  context?: {
    endpoint?: string;
    method?: string;
    userId?: string;
    requestId?: string;
  }
): void {
  const metric: ErrorMetric = {
    timestamp: new Date().toISOString(),
    code: (error as Error & { code?: string }).code || 'UNKNOWN_ERROR',
    message: error.message,
    stack: error.stack,
    ...context
  };
  
  errorMetrics.recordError(metric);
}
