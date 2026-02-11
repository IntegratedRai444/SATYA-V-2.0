/**
 * Structured Logging Utility
 * Provides standardized logging with context and correlation IDs
 */

import { logger } from '../config/logger';
import { randomUUID } from 'crypto';

export interface LogContext {
  requestId?: string;
  correlationId?: string;
  userId?: string;
  traceId?: string;
  spanId?: string;
}

export interface LogMetadata {
  method?: string;
  path?: string;
  statusCode?: number;
  duration?: number;
  userAgent?: string;
  ip?: string;
  [key: string]: unknown;
}

/**
 * Creates a correlation ID for request tracing
 */
export const createCorrelationId = (): string => {
  return `req_${Date.now()}_${randomUUID().substring(0, 8)}`;
};

/**
 * Logs incoming request with structured format
 */
export const logIncomingRequest = (method: string, path: string, context?: LogContext, metadata?: LogMetadata) => {
  logger.info('[ROUTE] Incoming request', {
    method,
    path,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Logs authentication verification
 */
export const logAuthVerification = (userId: string, success: boolean, context?: LogContext, metadata?: LogMetadata) => {
  logger.info(success ? '[AUTH] User verified' : '[AUTH] Verification failed', {
    userId,
    success,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Logs database operations
 */
export const logDatabaseOperation = (operation: string, table: string, success: boolean, context?: LogContext, metadata?: LogMetadata) => {
  logger.info(success ? `[DB] Query executed` : `[DB] Query failed`, {
    operation,
    table,
    success,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Logs ML service calls
 */
export const logMLServiceCall = (endpoint: string, success: boolean, context?: LogContext, metadata?: LogMetadata) => {
  logger.info(success ? '[ML] Service called' : '[ML] Service call failed', {
    endpoint,
    success,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Logs successful responses
 */
export const logSuccessResponse = (message: string, context?: LogContext, metadata?: LogMetadata) => {
  logger.info('[SUCCESS] Response sent', {
    message,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Logs detailed errors with context
 */
export const logDetailedError = (error: Error, context?: LogContext, metadata?: LogMetadata) => {
  logger.error('[ERROR] Detailed error', {
    message: error.message,
    stack: error.stack,
    timestamp: new Date().toISOString(),
    ...context,
    ...metadata
  });
};

/**
 * Creates request context for logging
 */
export const createRequestContext = (req: any): LogContext => {
  return {
    requestId: req.headers['x-request-id'] || randomUUID().substring(0, 8),
    correlationId: req.correlationId || createCorrelationId(),
    userId: req.user?.id,
    traceId: req.headers['x-trace-id'],
    spanId: req.headers['x-span-id']
  };
};
