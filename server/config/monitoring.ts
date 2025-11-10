import { createLogger, format, transports } from 'winston';
import { Request, Response } from 'express';
import { defaultSecurityConfig } from './security';

// Initialize logger
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.json()
  ),
  defaultMeta: { service: 'api-security' },
  transports: [
    new transports.Console(),
    new transports.File({ filename: 'logs/security.log', level: 'warn' })
  ]
});

// Track rate limit events
export const trackRateLimit = (req: Request, res: Response) => {
  const { method, path, ip } = req;
  const { limit, remaining, resetTime } = req.rateLimit || {};
  
  logger.warn('Rate limit warning', {
    type: 'RATE_LIMIT',
    method,
    path,
    ip,
    limit,
    remaining,
    resetTime: resetTime?.toISOString(),
    timestamp: new Date().toISOString(),
    userAgent: req.get('user-agent')
  });

  // Send metrics to monitoring system
  // Example: metrics.increment('rate_limit.hit', { path, method });
};

// Track validation failures
export const trackValidationError = (req: Request, errors: any[]) => {
  const { method, path, body } = req;
  
  logger.warn('Validation failed', {
    type: 'VALIDATION_ERROR',
    method,
    path,
    errors,
    timestamp: new Date().toISOString(),
    ip: req.ip,
    userAgent: req.get('user-agent'),
    // Redact sensitive data from logs
    body: redactSensitiveData(body)
  });
};

// Track API key usage
export const trackApiKeyUsage = (req: Request, key: string) => {
  const { method, path } = req;
  const maskedKey = maskApiKey(key);
  
  logger.info('API key used', {
    type: 'API_KEY_USAGE',
    method,
    path,
    key: maskedKey,
    timestamp: new Date().toISOString(),
    ip: req.ip,
    userAgent: req.get('user-agent')
  });

  // Update usage in database
  // Example: db.updateApiKeyUsage(key, { lastUsed: new Date() });
};

// Helper to mask sensitive data in logs
function redactSensitiveData(obj: any): any {
  if (!obj || typeof obj !== 'object') return obj;
  
  const sensitiveFields = [
    'password', 'apiKey', 'token', 'authorization',
    'creditCard', 'ssn', 'cvv', 'expiryDate'
  ];
  
  const result: any = Array.isArray(obj) ? [] : {};
  
  for (const [key, value] of Object.entries(obj)) {
    if (sensitiveFields.includes(key.toLowerCase())) {
      result[key] = '***REDACTED***';
    } else if (value && typeof value === 'object') {
      result[key] = redactSensitiveData(value);
    } else {
      result[key] = value;
    }
  }
  
  return result;
}

// Helper to mask API key
function maskApiKey(key: string, visibleChars = 4): string {
  if (!key) return '';
  if (key.length <= visibleChars * 2) return '****';
  return `${key.substring(0, visibleChars)}...${key.substring(key.length - visibleChars)}`;
}

// Initialize monitoring
const initMonitoring = () => {
  // Set up metrics collection
  // Example: metrics.init({ host: process.env.METRICS_HOST });
  
  // Log unhandled promise rejections
  process.on('unhandledRejection', (reason) => {
    logger.error('Unhandled Rejection:', { reason });
  });

  // Log uncaught exceptions
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', { error });
  });
};

export default {
  logger,
  trackRateLimit,
  trackValidationError,
  trackApiKeyUsage,
  initMonitoring
};
