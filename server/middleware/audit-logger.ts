import { Request, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';
import { AuthenticatedRequest } from '../types/auth';
import { logger } from '../config/logger';
import crypto from 'crypto';

// Types of actions to audit
type AuditAction = 
  | 'user_login'
  | 'user_logout'
  | 'user_register'
  | 'user_profile_update'
  | 'user_preferences_update'
  | 'user_delete'
  | 'analysis_create'
  | 'analysis_complete'
  | 'analysis_failed'
  | 'file_upload'
  | 'notification_create'
  | 'notification_read'
  | 'chat_message_send'
  | 'chat_conversation_create'
  | 'auth_attempt_success'
  | 'auth_attempt_failed'
  | 'admin_action'
  | 'sensitive_data_access';

// Risk scoring levels
type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

// Audit entry interface
interface AuditEntry {
  id: string;
  timestamp: string;
  userId?: string;
  action: AuditAction;
  resource?: string;
  method: string;
  url: string;
  ip: string;
  userAgent: string;
  statusCode: number;
  riskLevel: RiskLevel;
  responseTime: number;
  requestData?: Record<string, unknown>;
  responseData?: unknown;
  metadata: Record<string, unknown>;
}

// Audit logging middleware
export const auditLogger = (action: AuditAction, resource?: string) => {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const startTime = Date.now();
    
    // Store original res.json method
    const originalJson = res.json;
    
    // Override res.json to capture response data
    res.json = function(data: unknown) {
      // Log the audit entry after response is sent
      setTimeout(async () => {
        await logEnhancedAuditEntry(req, res, action, resource, data, startTime);
      }, 0);
      
      // Call original json method
      return originalJson.call(this, data);
    };
    
    next();
  };
};

// Risk scoring function
function calculateRiskLevel(action: AuditAction, statusCode: number): RiskLevel {
  // Critical actions
  if (action.includes('delete') || action.includes('admin')) {
    return 'critical';
  }
  
  // High risk actions
  if (action.includes('sensitive_data') || action.includes('auth_attempt_failed')) {
    return 'high';
  }
  
  // Medium risk actions
  if (action.includes('profile_update') || action.includes('file_upload')) {
    return 'medium';
  }
  
  // Failed requests
  if (statusCode >= 400) {
    return 'medium';
  }
  
  return 'low';
}

// Enhanced audit logging middleware with risk scoring
export const enhancedAuditLogger = (action: AuditAction, resource?: string) => {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const startTime = Date.now();
    
    // Store original res.json method
    const originalJson = res.json;
    
    // Override res.json to capture response data
    res.json = function(data: unknown) {
      // Log the audit entry after response is sent
      setTimeout(async () => {
        await logEnhancedAuditEntry(req, res, action, resource, data, startTime);
      }, 0);
      
      // Call original json method
      return originalJson.call(this, data);
    };
    
    next();
  };
};

// Function to log enhanced audit entries with risk scoring
async function logEnhancedAuditEntry(
  req: AuthenticatedRequest, 
  res: Response, 
  action: AuditAction, 
  resource?: string, 
  responseData?: unknown,
  startTime?: number
): Promise<void> {
  try {
    const userId = req.user?.id;
    const userEmail = req.user?.email;
    const responseTime = startTime ? Date.now() - startTime : 0;
    
    // Calculate risk level
    const riskLevel = calculateRiskLevel(action, res.statusCode);
    
    // Prepare enhanced audit data
    const auditEntry: AuditEntry = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      userId: userId || undefined,
      action,
      resource,
      method: req.method,
      url: req.url,
      ip: getClientIP(req),
      userAgent: req.get('User-Agent') || 'unknown',
      statusCode: res.statusCode,
      riskLevel,
      responseTime,
      requestData: sanitizeRequestData(req),
      responseData: responseData ? sanitizeResponseData(responseData) : undefined,
      metadata: {
        email: userEmail,
        requestId: (req as { id?: string }).id || generateRequestId(),
        sessionId: (req as { session?: { id?: string } }).session?.id,
        geoLocation: (req as { geo?: unknown }).geo,
        deviceFingerprint: generateDeviceFingerprint(req)
      }
    };
    
    // Log to console for immediate visibility
    logger.info('Audit Log', {
      action: auditEntry.action,
      userId: auditEntry.userId,
      riskLevel: auditEntry.riskLevel,
      ip: auditEntry.ip,
      responseTime: auditEntry.responseTime
    });
    
    // Insert audit log into database
    const { error } = await supabase
      .from('audit_logs')
      .insert(auditEntry);
    
    if (error) {
      logger.error('Failed to log audit entry:', error);
    }
    
    // Trigger alerts for high-risk activities
    if (riskLevel === 'critical' || riskLevel === 'high') {
      await triggerSecurityAlert(auditEntry);
    }
  } catch (error) {
    logger.error('Enhanced audit logging error:', error);
  }
}

// Helper function to get client IP
function getClientIP(req: Request): string {
  return (
    req.get('X-Forwarded-For')?.split(',')[0] ||
    req.get('X-Real-IP') ||
    req.get('X-Client-IP') ||
    req.connection?.remoteAddress ||
    req.socket?.remoteAddress ||
    'unknown'
  );
}

// Helper function to generate request ID
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Helper function to sanitize request data
function sanitizeRequestData(req: Request): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {
    method: req.method,
    url: req.url,
    query: req.query,
    params: req.params
  };
  
  // Remove sensitive data from request body
  if (req.body) {
    const bodyCopy = { ...req.body } as Record<string, unknown>;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { password, token, ...safeBody } = bodyCopy;
    sanitized.body = safeBody;
  }
  
  return sanitized;
}

// Helper function to sanitize response data
function sanitizeResponseData(data: unknown): unknown {
  if (!data || typeof data !== 'object') {
    return data;
  }
  
  // Remove sensitive fields from response
  const dataCopy = { ...data } as Record<string, unknown>;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { token: _token, password: _password, apiKey: _apiKey, ...safeData } = dataCopy;
  return safeData;
}

// Helper function to generate device fingerprint
function generateDeviceFingerprint(req: Request): string {
  const userAgent = req.get('User-Agent') || 'unknown';
  const acceptLanguage = req.get('Accept-Language') || 'unknown';
  const acceptEncoding = req.get('Accept-Encoding') || 'unknown';
  
  return crypto
    .createHash('sha256')
    .update(`${userAgent}-${acceptLanguage}-${acceptEncoding}`)
    .digest('hex')
    .substring(0, 16);
}

// Helper function to trigger security alerts
async function triggerSecurityAlert(auditEntry: AuditEntry): Promise<void> {
  try {
    logger.warn('SECURITY ALERT', {
      type: 'HIGH_RISK_ACTIVITY',
      action: auditEntry.action,
      userId: auditEntry.userId,
      ip: auditEntry.ip,
      riskLevel: auditEntry.riskLevel,
      timestamp: auditEntry.timestamp
    });
    
    // TODO: Integrate with alerting system (email, Slack, etc.)
    // await sendSecurityAlert(auditEntry);
  } catch (error) {
    logger.error('Failed to trigger security alert:', error);
  }
}


// Specific audit middleware for common actions
export const auditAuth = {
  login: auditLogger('user_login', 'auth'),
  logout: auditLogger('user_logout', 'auth'),
  register: auditLogger('user_register', 'auth'),
  attemptSuccess: auditLogger('auth_attempt_success', 'auth'),
  attemptFailed: auditLogger('auth_attempt_failed', 'auth')
};

export const auditUser = {
  profileUpdate: auditLogger('user_profile_update', 'user'),
  preferencesUpdate: auditLogger('user_preferences_update', 'user'),
  delete: auditLogger('user_delete', 'user')
};

export const auditAnalysis = {
  create: auditLogger('analysis_create', 'analysis'),
  complete: auditLogger('analysis_complete', 'analysis'),
  failed: auditLogger('analysis_failed', 'analysis'),
  fileUpload: auditLogger('file_upload', 'file')
};

export const auditChat = {
  messageSend: auditLogger('chat_message_send', 'chat'),
  conversationCreate: auditLogger('chat_conversation_create', 'chat')
};

export const auditNotification = {
  create: auditLogger('notification_create', 'notification'),
  read: auditLogger('notification_read', 'notification')
};

// Generic audit middleware for admin actions
export const auditAdmin = auditLogger('admin_action', 'admin');

// Audit middleware for sensitive data access
export const auditSensitiveAccess = auditLogger('sensitive_data_access', 'data');

// Function to create custom audit middleware
export const createAuditMiddleware = (action: AuditAction, resource?: string) => {
  return auditLogger(action, resource);
};
