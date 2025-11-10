import { logger } from '../config/logger';
import fs from 'fs/promises';
import path from 'path';

export interface AuditEvent {
  id: string;
  timestamp: Date;
  userId?: string;
  sessionId?: string;
  action: string;
  resource: string;
  details: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
  success: boolean;
  errorMessage?: string;
  duration?: number;
  metadata?: Record<string, any>;
}

export interface SecurityEvent extends AuditEvent {
  severity: 'low' | 'medium' | 'high' | 'critical';
  threatType: 'authentication' | 'authorization' | 'input_validation' | 'rate_limiting' | 'suspicious_activity';
  blocked: boolean;
}

class AuditLogger {
  private auditLogPath: string;
  private securityLogPath: string;
  private maxLogFileSize: number = 10 * 1024 * 1024; // 10MB
  private maxLogFiles: number = 10;

  constructor() {
    const logDir = process.env.LOG_DIR || './logs';
    this.auditLogPath = path.join(logDir, 'audit.log');
    this.securityLogPath = path.join(logDir, 'security.log');
    this.ensureLogDirectory();
  }

  private async ensureLogDirectory(): Promise<void> {
    try {
      const logDir = path.dirname(this.auditLogPath);
      await fs.mkdir(logDir, { recursive: true });
    } catch (error) {
      logger.error('Failed to create log directory:', error);
    }
  }

  // General audit logging
  async logEvent(event: Omit<AuditEvent, 'id' | 'timestamp'>): Promise<void> {
    const auditEvent: AuditEvent = {
      id: this.generateEventId(),
      timestamp: new Date(),
      ...event
    };

    try {
      // Log to main logger
      logger.info('Audit Event', auditEvent);

      // Write to audit log file
      await this.writeToLogFile(this.auditLogPath, auditEvent);

      // If it's a failed action, also log as potential security event
      if (!event.success) {
        await this.logSecurityEvent({
          ...auditEvent,
          severity: 'medium',
          threatType: 'suspicious_activity',
          blocked: false
        });
      }
    } catch (error) {
      logger.error('Failed to log audit event:', error);
    }
  }

  // Security-specific logging
  async logSecurityEvent(event: Omit<SecurityEvent, 'id' | 'timestamp'>): Promise<void> {
    const securityEvent: SecurityEvent = {
      id: this.generateEventId(),
      timestamp: new Date(),
      ...event
    };

    try {
      // Log to main logger with appropriate level
      const logLevel = this.getLogLevel(securityEvent.severity);
      logger[logLevel]('Security Event', securityEvent);

      // Write to security log file
      await this.writeToLogFile(this.securityLogPath, securityEvent);

      // For critical events, also trigger alerts
      if (securityEvent.severity === 'critical') {
        // This would integrate with the alerting system
        logger.error(`CRITICAL SECURITY EVENT: ${securityEvent.action}`, securityEvent);
      }
    } catch (error) {
      logger.error('Failed to log security event:', error);
    }
  }

  // Specific audit methods for common actions
  async logAuthentication(userId: string, success: boolean, details: Record<string, any>, req?: any): Promise<void> {
    await this.logEvent({
      userId,
      action: 'authentication',
      resource: 'auth',
      details,
      success,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent'),
      errorMessage: success ? undefined : details.error
    });

    // Log security event for failed authentication
    if (!success) {
      await this.logSecurityEvent({
        userId,
        action: 'failed_authentication',
        resource: 'auth',
        details,
        success: false,
        severity: 'medium',
        threatType: 'authentication',
        blocked: false,
        ipAddress: req?.ip,
        userAgent: req?.get('User-Agent')
      });
    }
  }

  async logAnalysis(userId: string, analysisType: string, success: boolean, details: Record<string, any>, duration?: number, req?: any): Promise<void> {
    await this.logEvent({
      userId,
      action: 'analysis',
      resource: `analysis_${analysisType}`,
      details: {
        ...details,
        analysisType
      },
      success,
      duration,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent'),
      errorMessage: success ? undefined : details.error
    });
  }

  async logFileUpload(userId: string, fileName: string, fileSize: number, success: boolean, details: Record<string, any>, req?: any): Promise<void> {
    await this.logEvent({
      userId,
      action: 'file_upload',
      resource: 'file',
      details: {
        ...details,
        fileName,
        fileSize
      },
      success,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent'),
      errorMessage: success ? undefined : details.error
    });
  }

  async logSessionActivity(userId: string, sessionId: string, action: string, success: boolean, details: Record<string, any>, req?: any): Promise<void> {
    await this.logEvent({
      userId,
      sessionId,
      action: `session_${action}`,
      resource: 'session',
      details,
      success,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent'),
      errorMessage: success ? undefined : details.error
    });
  }

  async logApiAccess(userId: string, endpoint: string, method: string, statusCode: number, duration: number, req?: any): Promise<void> {
    await this.logEvent({
      userId,
      action: 'api_access',
      resource: endpoint,
      details: {
        method,
        statusCode,
        endpoint
      },
      success: statusCode < 400,
      duration,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent')
    });
  }

  async logRateLimitViolation(ipAddress: string, endpoint: string, details: Record<string, any>, req?: any): Promise<void> {
    await this.logSecurityEvent({
      action: 'rate_limit_violation',
      resource: endpoint,
      details,
      success: false,
      severity: 'medium',
      threatType: 'rate_limiting',
      blocked: true,
      ipAddress,
      userAgent: req?.get('User-Agent')
    });
  }

  async logSuspiciousActivity(userId: string | undefined, activity: string, details: Record<string, any>, severity: SecurityEvent['severity'], req?: any): Promise<void> {
    await this.logSecurityEvent({
      userId,
      action: 'suspicious_activity',
      resource: 'system',
      details: {
        ...details,
        activity
      },
      success: false,
      severity,
      threatType: 'suspicious_activity',
      blocked: false,
      ipAddress: req?.ip,
      userAgent: req?.get('User-Agent')
    });
  }

  // Query methods for audit trails
  async getAuditTrail(userId?: string, startDate?: Date, endDate?: Date, action?: string): Promise<AuditEvent[]> {
    // In a real implementation, this would query a database
    // For now, we'll return a placeholder
    logger.info('Audit trail requested', { userId, startDate, endDate, action });
    return [];
  }

  async getSecurityEvents(severity?: SecurityEvent['severity'], startDate?: Date, endDate?: Date): Promise<SecurityEvent[]> {
    // In a real implementation, this would query a database
    logger.info('Security events requested', { severity, startDate, endDate });
    return [];
  }

  // Utility methods
  private generateEventId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private getLogLevel(severity: SecurityEvent['severity']): 'info' | 'warn' | 'error' {
    switch (severity) {
      case 'low': return 'info';
      case 'medium': return 'warn';
      case 'high':
      case 'critical': return 'error';
      default: return 'info';
    }
  }

  private async writeToLogFile(filePath: string, event: AuditEvent | SecurityEvent): Promise<void> {
    try {
      const logLine = JSON.stringify(event) + '\n';
      
      // Check file size and rotate if necessary
      await this.rotateLogIfNeeded(filePath);
      
      // Append to log file
      await fs.appendFile(filePath, logLine, 'utf8');
    } catch (error) {
      logger.error(`Failed to write to log file ${filePath}:`, error);
    }
  }

  private async rotateLogIfNeeded(filePath: string): Promise<void> {
    try {
      const stats = await fs.stat(filePath);
      
      if (stats.size > this.maxLogFileSize) {
        // Rotate logs
        for (let i = this.maxLogFiles - 1; i > 0; i--) {
          const oldFile = `${filePath}.${i}`;
          const newFile = `${filePath}.${i + 1}`;
          
          try {
            await fs.rename(oldFile, newFile);
          } catch (error) {
            // File might not exist, which is fine
          }
        }
        
        // Move current log to .1
        await fs.rename(filePath, `${filePath}.1`);
      }
    } catch (error) {
      // File might not exist yet, which is fine
    }
  }

  // Express middleware for automatic API logging
  auditMiddleware() {
    return (req: any, res: any, next: any) => {
      const startTime = Date.now();
      
      // Override res.end to capture response
      const originalEnd = res.end;
      res.end = async function(this: any, ...args: any[]) {
        const duration = Date.now() - startTime;
        
        // Log API access
        await auditLogger.logApiAccess(
          req.user?.user_id,
          req.path,
          req.method,
          res.statusCode,
          duration,
          req
        );
        
        // Call original end
        originalEnd.apply(this, args);
      };
      
      next();
    };
  }

  // Health check
  async getHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    auditLogWritable: boolean;
    securityLogWritable: boolean;
    lastWrite: Date;
  }> {
    let auditLogWritable = false;
    let securityLogWritable = false;
    
    try {
      await fs.access(path.dirname(this.auditLogPath), fs.constants.W_OK);
      auditLogWritable = true;
    } catch (error) {
      // Directory not writable
    }
    
    try {
      await fs.access(path.dirname(this.securityLogPath), fs.constants.W_OK);
      securityLogWritable = true;
    } catch (error) {
      // Directory not writable
    }
    
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    if (!auditLogWritable || !securityLogWritable) {
      status = 'degraded';
    }
    
    if (!auditLogWritable && !securityLogWritable) {
      status = 'unhealthy';
    }
    
    return {
      status,
      auditLogWritable,
      securityLogWritable,
      lastWrite: new Date()
    };
  }
}

// Singleton instance
const auditLogger = new AuditLogger();

export default auditLogger;