import * as winston from 'winston';
import * as path from 'path';
import * as fs from 'fs';
import { config } from './environment';
import type { Request, Response, NextFunction } from 'express';

// Define log levels
const logLevels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
};

// Define log colors
const logColors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  debug: 'blue',
};

winston.addColors(logColors);

// Secret redaction patterns
const secretPatterns = [
  /password[:=]\s*[^\s]+/gi,
  /secret[:=]\s*[^\s]+/gi,
  /key[:=]\s*[^\s]+/gi,
  /token[:=]\s*[^\s]+/gi,
  /auth[:=]\s*[^\s]+/gi,
  /supabase.*service.*key/gi,
  /database.*url/gi,
  /jwt.*secret/gi,
  /csrf.*token/gi,
  /openai.*key/gi,
  /sk-[a-zA-Z0-9]{20,}/gi
];

// Redact sensitive information from log messages
function redactSecrets(message: unknown): string {
  const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
  return messageStr.replace(new RegExp(secretPatterns.map((pattern) => pattern.source).join('|'), 'gi'), '[REDACTED]');
}

// Create custom format for development
const developmentFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.colorize({ all: true }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    const { requestId, traceId, spanId, ...restMeta } = meta;
    const traceInfo = requestId ? `[${requestId}|${traceId}|${spanId}]` : '';
    const metaStr = Object.keys(restMeta).length ? JSON.stringify(restMeta, null, 2) : '';
    const redactedMessage = redactSecrets(message);
    const redactedMeta = redactSecrets(metaStr);
    return `${timestamp} ${traceInfo} [${level}]: ${redactedMessage} ${redactedMeta}`.trim();
  })
);

// Create custom format for production
const productionFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format((info) => {
    // Add tracing context to all log entries
    const { requestId, traceId, spanId, ...rest } = info;
    const result = { ...rest };
    if (requestId) (result as Record<string, unknown>).requestId = requestId;
    if (traceId) (result as Record<string, unknown>).traceId = traceId;
    if (spanId) (result as Record<string, unknown>).spanId = spanId;
    return result;
  })(),
  winston.format.json()
);

// Create transports based on environment
function createTransports(): winston.transport[] {
  const transports: winston.transport[] = [];

  // Console transport
  transports.push(
    new winston.transports.Console({
      level: config.LOG_LEVEL,
      format: config.NODE_ENV === 'production' ? productionFormat : developmentFormat,
    })
  );

  // File transport for production or if LOG_FILE is specified
  if (config.NODE_ENV === 'production' || config.LOG_FILE) {
    const logDir = path.dirname(config.LOG_FILE || './logs/app.log');
    const logFile = config.LOG_FILE || './logs/app.log';

    // Ensure log directory exists
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    // Error log file
    transports.push(
      new winston.transports.File({
        filename: path.join(logDir, 'error.log'),
        level: 'error',
        format: productionFormat,
        maxsize: 5242880, // 5MB
        maxFiles: 5,
      })
    );

    // Combined log file
    transports.push(
      new winston.transports.File({
        filename: logFile,
        format: productionFormat,
        maxsize: 5242880, // 5MB
        maxFiles: 5,
      })
    );
  }

  return transports;
}

// Create logger instance
export const logger = winston.createLogger({
  levels: logLevels,
  level: config.LOG_LEVEL,
  format: config.NODE_ENV === 'production' ? productionFormat : developmentFormat,
  transports: createTransports(),
  exitOnError: false,
});

// Create request logger middleware
export function createRequestLogger() {
  return (req: Request, res: Response, next: NextFunction) => {
    const start = Date.now();
    
    // Log request
    logger.info('HTTP Request', {
      method: req.method,
      url: req.url,
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      timestamp: new Date().toISOString(),
    });

    // Log response when finished
    res.on('finish', () => {
      const duration = Date.now() - start;
      const logLevel = res.statusCode >= 400 ? 'error' : 'info';
      
      logger.log(logLevel, 'HTTP Response', {
        method: req.method,
        url: req.url,
        statusCode: res.statusCode,
        duration: `${duration}ms`,
        ip: req.ip,
        timestamp: new Date().toISOString(),
      });
    });

    next();
  };
}

// Create error logger
export function logError(error: Error, context?: Record<string, unknown>) {
  logger.error('Application Error', {
    message: error.message,
    stack: error.stack,
    name: error.name,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create performance logger
export function logPerformance(operation: string, duration: number, context?: Record<string, unknown>) {
  logger.info('Performance Metric', {
    operation,
    duration: `${duration}ms`,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create security logger
export function logSecurity(event: string, context?: Record<string, unknown>) {
  logger.warn('Security Event', {
    event,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create database logger
export function logDatabase(operation: string, context?: Record<string, unknown>) {
  logger.debug('Database Operation', {
    operation,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create AI engine logger
export function logAIEngine(operation: string, context?: Record<string, unknown>) {
  logger.info('AI Engine Operation', {
    operation,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create analysis-specific logger
export function logAnalysis(
  event: 'request' | 'start' | 'progress' | 'complete' | 'error' | 'timeout',
  context: {
    userId?: number;
    jobId?: string;
    analysisType?: string;
    filename?: string;
    fileSize?: number;
    stage?: string;
    percentage?: number;
    duration?: number;
    error?: string;
    result?: {
      authenticity?: string;
      confidence?: number;
      success?: boolean;
    };
    pythonServer?: {
      status?: string;
      responseTime?: number;
      retryAttempt?: number;
    };
    [key: string]: unknown;
  }
) {
  const logLevel = event === 'error' ? 'error' : event === 'timeout' ? 'warn' : 'info';
  
  logger.log(logLevel, `Analysis ${event.toUpperCase()}`, {
    event,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create Python bridge communication logger
export function logPythonBridge(
  operation: 'request' | 'response' | 'error' | 'retry' | 'timeout' | 'circuit_breaker',
  context: {
    endpoint?: string;
    method?: string;
    statusCode?: number;
    responseTime?: number;
    retryAttempt?: number;
    totalAttempts?: number;
    error?: string;
    circuitBreakerState?: string;
    requestId?: string;
    fileSize?: number;
    [key: string]: unknown;
  }
) {
  const logLevel = operation === 'error' || operation === 'timeout' ? 'error' : 
                   operation === 'retry' || operation === 'circuit_breaker' ? 'warn' : 'info';
  
  logger.log(logLevel, `Python Bridge ${operation.toUpperCase()}`, {
    operation,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create file processing logger
export function logFileProcessing(
  event: 'upload' | 'validation' | 'processing' | 'cleanup' | 'error',
  context: {
    userId?: number;
    filename?: string;
    fileSize?: number;
    mimeType?: string;
    uploadPath?: string;
    processingTime?: number;
    error?: string;
    [key: string]: unknown;
  }
) {
  const logLevel = event === 'error' ? 'error' : 'info';
  
  logger.log(logLevel, `File Processing ${event.toUpperCase()}`, {
    event,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create system health logger
export function logSystemHealth(
  component: 'python_server' | 'database' | 'websocket' | 'file_system' | 'memory',
  status: 'healthy' | 'degraded' | 'unhealthy',
  context?: {
    responseTime?: number;
    errorRate?: number;
    memoryUsage?: number;
    diskUsage?: number;
    connectionCount?: number;
    error?: string;
    [key: string]: unknown;
  }
) {
  const logLevel = status === 'unhealthy' ? 'error' : status === 'degraded' ? 'warn' : 'info';
  
  logger.log(logLevel, `System Health ${status.toUpperCase()}`, {
    component,
    status,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Create user activity logger
export function logUserActivity(
  action: 'login' | 'logout' | 'upload' | 'analysis_request' | 'download_report',
  context: {
    userId?: number;
    username?: string;
    ip?: string;
    userAgent?: string;
    sessionId?: string;
    analysisType?: string;
    filename?: string;
    [key: string]: unknown;
  }
) {
  logger.info(`User Activity ${action.toUpperCase()}`, {
    action,
    ...context,
    timestamp: new Date().toISOString(),
  });
}

// Export logger for direct use
export default logger;