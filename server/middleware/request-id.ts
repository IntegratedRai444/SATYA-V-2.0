import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

// Extend Request interface to include our custom properties
export interface RequestWithId extends Request {
  id: string;
  correlationId: string;
  startTime: number;
}

// Extend Response interface to include request ID
export interface ResponseWithId extends Response {
  requestId?: string;
}

// Request ID middleware
export const requestIdMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const requestWithId = req as RequestWithId;
  const responseWithId = res as ResponseWithId;
  
  // Generate unique request ID
  requestWithId.id = randomUUID();
  requestWithId.correlationId = requestWithId.id;
  requestWithId.startTime = Date.now();
  
  // Add request ID to response headers
  responseWithId.setHeader('X-Request-ID', requestWithId.id);
  responseWithId.setHeader('X-Correlation-ID', requestWithId.correlationId);
  responseWithId.requestId = requestWithId.id;
  
  next();
};

// Helper function to get request ID from headers
export const getRequestId = (req: Request): string => {
  const requestWithId = req as RequestWithId;
  return requestWithId.id || req.headers['x-request-id'] as string || 'unknown';
};

// Helper function to get correlation ID from headers
export const getCorrelationId = (req: Request): string => {
  const requestWithId = req as RequestWithId;
  return requestWithId.correlationId || req.headers['x-correlation-id'] as string || 'unknown';
};

// Helper function to log with request context
export const logWithContext = (message: string, req: Request, additionalData: Record<string, unknown> = {}) => {
  const requestId = getRequestId(req);
  const correlationId = getCorrelationId(req);
  
  console.log(JSON.stringify({
    message,
    requestId,
    correlationId,
    timestamp: new Date().toISOString(),
    ...additionalData
  }));
};

// Helper to create response with correlation
export const createResponseWithCorrelation = (req: RequestWithId, res: Response, statusCode: number, data: unknown, message?: string) => {
  const response = {
    success: statusCode < 400,
    data,
    error: message || undefined,
    correlationId: req.correlationId,
    requestId: req.id,
    timestamp: new Date().toISOString()
  };
  
  return res.status(statusCode).json(response);
};
