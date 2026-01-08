import { Request, Response, NextFunction } from 'express';
import { metrics } from '../monitoring/metrics';
import { logger } from '../config/logger';

/**
 * Middleware to track HTTP request metrics
 * - Tracks request duration
 * - Counts total requests
 * - Tracks error rates
 */
export const httpMetrics = (req: Request, res: Response, next: NextFunction) => {
  const start = process.hrtime();
  const { method, path, originalUrl } = req;
  
  // Skip metrics endpoint to avoid noise
  if (path === '/metrics') {
    return next();
  }

  // Add response finish event listener
  res.on('finish', () => {
    try {
      const [seconds, nanoseconds] = process.hrtime(start);
      const duration = seconds + nanoseconds / 1e9;
      
      const statusCode = res.statusCode;
      const route = req.route?.path || path;
      
      // Record request duration
      metrics.http.requestDuration
        .labels(method, route, statusCode.toString())
        .observe(duration);
      
      // Count total requests
      metrics.http.requestsTotal
        .labels(method, route, statusCode.toString())
        .inc();
      
      // Log slow requests
      if (duration > 1) { // Log requests slower than 1 second
        logger.warn('Slow request detected', {
          method,
          url: originalUrl,
          duration: parseFloat(duration.toFixed(4)),
          statusCode,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      logger.error('Error in metrics middleware', {
        error: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Error handling
  const originalSend = res.send;
  res.send = function (body) {
    if (res.statusCode >= 400) {
      metrics.http.requestErrors
        .labels(method, path, res.statusCode.toString(), 'http_error')
        .inc();
      
      logger.error('HTTP error response', {
        statusCode: res.statusCode,
        method,
        url: originalUrl,
        timestamp: new Date().toISOString(),
        body: typeof body === 'string' ? body.substring(0, 500) : 'Non-string body'
      });
    }
    return originalSend.apply(res, arguments as any);
  };

  next();
};

/**
 * Middleware to track database query metrics
 */
export const dbMetrics = (collection: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const start = process.hrtime();
    const { method } = req;
    
    try {
      await next();
      
      const [seconds, nanoseconds] = process.hrtime(start);
      const duration = seconds + nanoseconds / 1e9;
      
      metrics.db.queryDuration
        .labels(method, collection, 'true')
        .observe(duration);
      
    } catch (error) {
      const [seconds, nanoseconds] = process.hrtime(start);
      const duration = seconds + nanoseconds / 1e9;
      
      metrics.db.queryDuration
        .labels(method, collection, 'false')
        .observe(duration);
      
      metrics.errors.inc({
        type: 'database_error',
        severity: 'high',
        component: 'database'
      });
      
      throw error; // Re-throw to be handled by error middleware
    }
  };
};

/**
 * Error tracking middleware
 */
export const errorMetrics = (error: Error, req: Request, res: Response, next: NextFunction) => {
  const { method, path, originalUrl } = req;
  
  // Log the error
  logger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    method,
    url: originalUrl,
    timestamp: new Date().toISOString()
  });
  
  // Increment error counter
  metrics.errors.inc({
    type: 'unhandled_error',
    severity: 'critical',
    component: 'api'
  });
  
  // Send error response
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong',
    ...(process.env.NODE_ENV === 'development' && { stack: error.stack })
  });
};
