/**
 * Error Tracking Service using Sentry
 * Captures and reports errors to Sentry for monitoring
 */

import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';
import { httpIntegration } from '@sentry/node';
import type { Scope, Span, SeverityLevel } from '@sentry/types';

// Extend Sentry types
declare module '@sentry/node' {
  interface RequestEventData {
    query?: Record<string, unknown>;
    params?: Record<string, string>;
    data?: unknown;
  }
}
import type { Express, NextFunction, Request, Response } from 'express';
import { logger } from '../config/logger';

// Type declarations for Express integration

// Type definitions for better type safety
type ErrorHandler = (error: Error, req: Request, res: Response, next: NextFunction) => void;
type RequestHandler = (req: Request, res: Response, next: NextFunction) => void;
type TracingHandler = (req: Request, res: Response, next: NextFunction) => void;

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
  private initSentry(app?: Express) {
    if (!this.config.enabled || !this.config.dsn) {
      logger.warn('Sentry is disabled. No DSN provided or error tracking is disabled.');
      return;
    }

    try {
      const integrations: any[] = [
        // Add profiling integration
        nodeProfilingIntegration(),
        // Add HTTP integration for request handling
        httpIntegration()
      ];

      Sentry.init({
        dsn: this.config.dsn,
        environment: this.config.environment,
        sampleRate: this.config.sampleRate,
        tracesSampleRate: this.config.tracesSampleRate,
        integrations: integrations,
        // Enable HTTP calls tracing
        tracePropagationTargets: ['localhost', /^https?:\/\//],
      });

      this.initialized = true;
      logger.info('Sentry initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Sentry', { error });
    }
  }

  /**
   * Initialize Sentry error tracking
   */
  initialize(app?: Express): void {
    if (!this.config.enabled) {
      logger.info('Error tracking is disabled');
      return;
    }

    this.initSentry(app);
  }

  /**
   * Get Sentry request handler middleware
   */
  getRequestHandler(): RequestHandler {
    if (!this.initialized) return (_req: Request, _res: Response, next: NextFunction) => next();
    
    return (req: Request, res: Response, next: NextFunction) => {
      // Start a new transaction for the request
      const transaction = Sentry.startInactiveSpan({
        name: `${req.method} ${req.path}`,
        op: 'http.server',
        attributes: {
          'http.method': req.method,
          'http.url': req.url,
          'http.route': req.route?.path || req.path
        }
      });

      // In Sentry v10, we set the transaction name and operation
      if (transaction) {
        const transactionName = `${req.method} ${req.path}`;
        transaction.updateName(transactionName);
        
        Sentry.setContext('transaction', {
          name: transactionName,
          op: 'http.server',
          status: 'ok'
        });
      }

      // Set user context if available
      if (req.user) {
        const user: any = req.user;
        Sentry.setUser({
          id: user.id?.toString(),
          email: user.email,
          // Only include username if it exists on the user object
          ...(user.username && { username: user.username })
        });
      }

      // Set request data (convert headers to a plain object with string values)
      const headers: Record<string, string> = {};
      Object.entries(req.headers).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          headers[key] = value.join(', ');
        } else if (value) {
          headers[key] = value;
        }
      });

      Sentry.setContext('request', {
        method: req.method,
        url: req.url,
        headers,
        query: req.query,
        params: req.params,
        body: req.body
      });

      // When the request is finished, finish the transaction
      res.on('finish', () => {
        if (transaction) {
          transaction.setAttribute('http.status_code', res.statusCode);
          transaction.end();
        }
      });

      next();
    };
  }

  getErrorHandler(): ErrorHandler {
    if (!this.initialized) {
      return (error: Error, _req: Request, _res: Response, next: NextFunction) => next(error);
    }
    
    return (error: Error, req: Request, _res: Response, next: NextFunction) => {
      if (!this.shouldHandleError(error)) {
        return next(error);
      }

      // Capture the exception with request context
      Sentry.withScope(scope => {
        // Add request data to the scope
        const headers: Record<string, string> = {};
        Object.entries(req.headers).forEach(([key, value]) => {
          if (Array.isArray(value)) {
            headers[key] = value.join(', ');
          } else if (value) {
            headers[key] = value;
          }
        });

        // Add request data to the scope
        scope.addEventProcessor((event) => {
          // Set basic request data
          event.request = {
            method: req.method,
            url: req.url,
            headers: headers as Record<string, string>,
            data: req.body
          };
          
          // Add additional context as tags
          if (Object.keys(req.query).length > 0) {
            scope.setContext('query', req.query);
          }
          if (Object.keys(req.params).length > 0) {
            scope.setContext('params', req.params);
          }
          
          return event;
        });
        
        // Add additional context
        scope.setContext('request', {
          method: req.method,
          url: req.url,
          headers: req.headers,
          query: req.query,
          params: req.params,
          body: req.body
        });

        // Capture the error
        Sentry.captureException(error);
      });

      next(error);
    };
  }

  private shouldHandleError(error: Error & { statusCode?: number }): boolean {
    // Capture all 500+ errors or any unhandled errors
    return (error.statusCode && error.statusCode >= 500) || !error.statusCode;
  }

  /**
   * Get Sentry tracing handler middleware
   */
  getTracingHandler(): TracingHandler {
    if (!this.initialized) return (_req: Request, _res: Response, next: NextFunction) => next();
    
    return (req: Request, res: Response, next: NextFunction) => {
      const url = req.originalUrl || req.url;
      
      // Don't trace health checks or static assets
      if (url === '/health' || url.match(/\.(css|js|jpg|png|svg|ico|woff2?)$/i)) {
        return next();
      }
      
      // The actual tracing is now handled by the httpIntegration in the init method
      next();
    };
  }

  /**
   * Capture an exception
   */
  captureException(error: Error, context?: Record<string, unknown>): void {
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
  captureMessage(message: string, level: SeverityLevel = 'info', context?: Record<string, unknown>): void {
    if (!this.initialized) return;
    
    const sentryLevel = level as SeverityLevel;
    
    Sentry.withScope((scope) => {
      if (context) {
        scope.setExtras(context);
      }
      scope.setLevel(sentryLevel);
      Sentry.captureMessage(message);
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
  addBreadcrumb(message: string, category: string, data?: Record<string, unknown>): void {
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
    
    const transaction = Sentry.startInactiveSpan({
      name: name || 'unnamed-transaction',
      op: op || 'transaction',
      attributes: {
        startTime: new Date().toISOString(),
        'transaction.status': 'ok'
      }
    });
    
    // No need to manually set the span in the scope as it's handled by Sentry
    
    return transaction;
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
