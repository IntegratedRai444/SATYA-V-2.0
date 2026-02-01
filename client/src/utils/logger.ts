/**
 * Centralized logging utility
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogContext {
  [key: string]: any;
}

/**
 * Log a message with the specified level
 */
const log = (level: LogLevel, message: string, ...args: any[]) => {
  const timestamp = new Date().toISOString();
  const context = args.reduce((acc, arg) => {
    if (typeof arg === 'object' && arg !== null) {
      return { ...acc, ...arg };
    }
    return acc;
  }, {});

  const logEntry = {
    timestamp,
    level,
    message,
    ...context,
  };

  // In development, log to console with colors
  if (process.env.NODE_ENV === 'development') {
    const colors = {
      debug: '\x1b[36m', // Cyan
      info: '\x1b[32m', // Green
      warn: '\x1b[33m', // Yellow
      error: '\x1b[31m', // Red
      reset: '\x1b[0m', // Reset
    };

    console[level](
      `${colors[level]}[${level.toUpperCase()}]${colors.reset} ${message}`,
      Object.keys(context).length > 0 ? context : ''
    );
  }

  // In production, logs are sent to console and can be integrated with external logging services
  if (process.env.NODE_ENV === 'production') {
    // Production logging is handled by console output
    // External logging service integration can be added when needed
  }

  return logEntry;
};

/**
 * Log a debug message
 */
const debug = (message: string, context?: LogContext) =>
  log('debug', message, context);

/**
 * Log an info message
 */
const info = (message: string, context?: LogContext) =>
  log('info', message, context);

/**
 * Log a warning message
 */
const warn = (message: string, context?: LogContext) =>
  log('warn', message, context);

/**
 * Log an error message
 */
const error = (message: string, error?: Error, context?: LogContext) => {
  const errorContext = error
    ? {
        error: {
          name: error.name,
          message: error.message,
          stack: error.stack,
        },
        ...context,
      }
    : context;
  return log('error', message, errorContext);
};

const logger = {
  debug,
  info,
  warn,
  error,
};

export default logger;
