/**
 * Centralized logging utility for the SatyaAI application
 * Provides environment-aware logging with support for different log levels
 * and optional remote error tracking integration
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LoggerConfig {
  level: LogLevel;
  enableConsole: boolean;
  enableRemote: boolean;
  remoteEndpoint?: string;
  environment: 'development' | 'production' | 'test';
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  context?: Record<string, any>;
  stack?: string;
  environment: string;
}

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

class Logger {
  private config: LoggerConfig;
  private logBuffer: LogEntry[] = [];
  private readonly MAX_BUFFER_SIZE = 100;

  constructor(config?: Partial<LoggerConfig>) {
    const isDev = import.meta.env.DEV;
    const environment = import.meta.env.MODE as 'development' | 'production' | 'test';
    
    this.config = {
      level: config?.level || (isDev ? 'debug' : 'error'),
      enableConsole: config?.enableConsole ?? isDev,
      enableRemote: config?.enableRemote ?? !isDev,
      remoteEndpoint: config?.remoteEndpoint,
      environment,
    };
  }

  /**
   * Check if a log level should be logged based on current configuration
   */
  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[this.config.level];
  }

  /**
   * Create a log entry object
   */

  private createLogEntry(
    level: LogLevel,
    message: string,
    context?: any,
    error?: Error
  ): LogEntry {
    return {
      level,
      message,
      timestamp: new Date().toISOString(),
      context: context ? this.sanitizeContext(context) : undefined,
      stack: error?.stack,
      environment: this.config.environment,
    };
  }

  /**
   * Sanitize context to remove sensitive information
   */
  private sanitizeContext(context: any): Record<string, any> {
    if (typeof context !== 'object' || context === null) {
      return { value: context };
    }

    const sanitized: Record<string, any> = {};
    const sensitiveKeys = ['password', 'token', 'apiKey', 'secret', 'authorization'];

    for (const [key, value] of Object.entries(context)) {
      const lowerKey = key.toLowerCase();
      if (sensitiveKeys.some(sensitive => lowerKey.includes(sensitive))) {
        sanitized[key] = '[REDACTED]';
      } else if (typeof value === 'object' && value !== null) {
        sanitized[key] = this.sanitizeContext(value);
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }

  /**
   * Log to console if enabled
   */
  private logToConsole(entry: LogEntry): void {
    if (!this.config.enableConsole) return;

    const prefix = `[${entry.level.toUpperCase()}] ${entry.timestamp}`;
    const message = `${prefix} - ${entry.message}`;

    switch (entry.level) {
      case 'debug':
        console.debug(message, entry.context || '');
        break;
      case 'info':
        console.info(message, entry.context || '');
        break;
      case 'warn':
        console.warn(message, entry.context || '');
        break;
      case 'error':
        console.error(message, entry.context || '', entry.stack || '');
        break;
    }
  }

  /**
   * Send log to remote endpoint
   */
  private async logToRemote(entry: LogEntry): Promise<void> {
    if (!this.config.enableRemote || !this.config.remoteEndpoint) return;

    try {
      await fetch(this.config.remoteEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(entry),
      });
    } catch (error) {
      // Silently fail to avoid infinite loops
      if (this.config.enableConsole) {
        console.error('Failed to send log to remote endpoint:', error);
      }
    }
  }

  /**
   * Add log entry to buffer
   */
  private addToBuffer(entry: LogEntry): void {
    this.logBuffer.push(entry);
    
    // Keep buffer size under control
    if (this.logBuffer.length > this.MAX_BUFFER_SIZE) {
      this.logBuffer.shift();
    }
  }

  /**
   * Core logging method
   */
  private log(level: LogLevel, message: string, context?: any, error?: Error): void {
    if (!this.shouldLog(level)) return;

    const entry = this.createLogEntry(level, message, context, error);
    
    this.addToBuffer(entry);
    this.logToConsole(entry);
    
    // Send to remote asynchronously (don't await)
    if (this.config.enableRemote) {
      this.logToRemote(entry).catch(() => {
        // Silently fail
      });
    }
  }

  /**
   * Log debug message
   */
  debug(message: string, context?: any): void {
    this.log('debug', message, context);
  }

  /**
   * Log info message
   */
  info(message: string, context?: any): void {
    this.log('info', message, context);
  }

  /**
   * Log warning message
   */
  warn(message: string, context?: any): void {
    this.log('warn', message, context);
  }

  /**
   * Log error message
   */
  error(message: string, error?: Error, context?: any): void {
    this.log('error', message, context, error);
  }

  /**
   * Get recent logs from buffer
   */
  getRecentLogs(count?: number): LogEntry[] {
    return count 
      ? this.logBuffer.slice(-count)
      : [...this.logBuffer];
  }

  /**
   * Clear log buffer
   */
  clearBuffer(): void {
    this.logBuffer = [];
  }

  /**
   * Update logger configuration
   */
  updateConfig(config: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<LoggerConfig> {
    return { ...this.config };
  }
}

// Create singleton instance
const logger = new Logger();

// Export singleton instance as default
export default logger;

// Export class for testing
export { Logger };
