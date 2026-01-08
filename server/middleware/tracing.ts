import { v4 as uuidv4 } from 'uuid';
import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

declare global {
  namespace Express {
    interface Request {
      requestId: string;
      traceId: string;
      spanId: string;
      parentSpanId?: string;
    }
  }
}

export function tracingMiddleware() {
  return (req: Request, res: Response, next: NextFunction) => {
    // Generate or extract trace context from headers
    const traceId = req.headers['x-trace-id']?.toString() || uuidv4();
    const spanId = uuidv4();
    const parentSpanId = req.headers['x-parent-span-id']?.toString();
    const requestId = req.headers['x-request-id']?.toString() || uuidv4();

    // Add to request object
    req.traceId = traceId;
    req.spanId = spanId;
    req.requestId = requestId;
    if (parentSpanId) {
      req.parentSpanId = parentSpanId;
    }

    // Set response headers
    res.setHeader('X-Request-ID', requestId);
    res.setHeader('X-Trace-ID', traceId);
    res.setHeader('X-Span-ID', spanId);

    // Log request start
    logger.info('Request started', {
      requestId,
      traceId,
      spanId,
      parentSpanId,
      method: req.method,
      path: req.path,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    });

    // Add response finish logging
    const start = Date.now();
    res.on('finish', () => {
      const duration = Date.now() - start;
      logger.info('Request completed', {
        requestId,
        traceId,
        statusCode: res.statusCode,
        duration,
        contentLength: res.get('content-length'),
      });
    });

    next();
  };
}

export function traceContext(req: Request) {
  return {
    requestId: req.requestId,
    traceId: req.traceId,
    spanId: req.spanId,
    parentSpanId: req.parentSpanId,
  };
}
