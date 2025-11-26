/**
 * Error Tracking Service using Sentry
 * Captures and reports errors to Sentry for monitoring
 */

import * as Sentry from '@sentry/node';
import { ProfilingIntegration } from '@sentry/profiling-node';
import type { Express } from 'express';
import { logger } from '../config/logger';

interface ErrorTrackingConfig {
  dsn?: string;
  environment: string;
  enabled: boolean;
  sampleRate: number;
  tracesSampleRate: number;
}

class ErrorTrackingService {
  private initialized = false;
  private config: ErrorTrackingConfig;

  constructor() {
    this.config = {
      dsn: process.env.SENTRY_DSN,
      environment: process.env.NODE_ENV || 'development',
      enabled: process.env.ENABLE_ERROR_TRACKING === 'true',
      sampleRate: parseFloat(process.env.SENTRY_SAMPLE_RATE || '1.0'),
      tracesSampleRate: parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || '0.1'),
    };
  }

  /**
   * Initialize Sentry error tracking
   */
  initialize(app?: Express): void {
    if (!this.config.enabled) {
      logger.info('Error tracking is disabled');
      return;
    }

    if (!this.config.dsn) {
      logger.warn('Sentry DSN not configured - error tracking disabled');
      return;
    }

    try {
      Sentry.init({
        dsn: this.config.dsn,
        environment: this.config.environment,
        integrations: [
          new Sentry.Integrations.Http({ tracing: true }),
          new Sentry.Integrations.Express({ app }),
          new ProfilingIntegration(),
        ],
        tracesSampleRate: this.config.tracesSampleRate,
        profilesSampleRate: this.config.tracesSampleRate,
        sampleRate: this.config.sampleRate,
        beforeSend(event, hint) {
          // Filter out sensitive data
          if (event.request) {
            delete event.request.cookies;
            if (event.request.headers) {
              delete event.request.headers.authorization;
              delete event.request.headers.cookie;
            }
          }
          return event;
        },
      });

      this.initialized = true;
      logger.info('✓ Sentry error tracking initialized', {
        environment: this.config.environment,
        tracesSampleRate: this.config.tracesSampleRate,
      });
    } catch (error) {
      logger.error('Failed to initialize Sentry', error as Error);
    }
  }

  /**
   * Get Sentry request handler middleware
   */
  getRequestHandler() {
    if (!this.initialized) {
      return (req: any, res: any, next: any) => next();
    }
    return Sentry.Handlers.requestHandler();
  }

  /**
   * Get Sentry tracing handler middleware
   */
  getTracingHandler() {
    if (!this.initialized) {
      return (req: any, res: any, next: any) => next();
    }
    return Sentry.Handlers.tracingHandler();
  }

  /**
   * Get Sentry error handler middleware
   */
  getErrorHandler() {
    if (!this.initialized) {
      return (err: any, req: any, res: any, next: any) => next(err);
    }
    return Sentry.Handlers.errorHandler();
  }

  /**
   * Capture an exception
   */
  captureException(error: Error, context?: Record<string, any>): void {
    if (!this.initialized) {
      logger.error('Error (Sentry disabled)', error, context);
      return;
    }

    Sentry.captureException(error, {
      extra: context,
    });
  }

  /**
   * Capture a message
   */
  captureMessage(message: string, level: Sentry.SeverityLevel = 'info', context?: Record<string, any>): void {
    if (!this.initialized) {
      logger.info(message, context);
      return;
    }

    Sentry.captureMessage(message, {
      level,
      extra: context,
    });
  }

  /**
   * Set user context
   */
  setUser(user: { id: string | number; email?: string; username?: string }): void {
    if (!this.initialized) return;

    Sentry.setUser({
      id: user.id.toString(),
      email: user.email,
      username: user.username,
    });
  }

  /**
   * Clear user context
   */
  clearUser(): void {
    if (!this.initialized) return;
    Sentry.setUser(null);
  }

  /**
   * Add breadcrumb for debugging
   */
  addBreadcrumb(message: string, category: string, data?: Record<string, any>): void {
    if (!this.initialized) return;

    Sentry.addBreadcrumb({
      message,
      category,
      data,
      level: 'info',
    });
  }

  /**
   * Start a transaction for performance monitoring
   */
  startTransaction(name: string, op: string): any {
    if (!this.initialized) return null;

    return Sentry.startTransaction({
      name,
      op,
    });
  }

  /**
   * Check if error tracking is enabled
   */
  isEnabled(): boolean {
    return this.initialized;
  }
}

// Export singleton instance
export const errorTracking = new ErrorTrackingService();
export default errorTracking;
