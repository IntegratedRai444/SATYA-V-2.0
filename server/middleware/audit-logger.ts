import { Request, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';
import { AuthenticatedRequest } from '../types/auth';

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

// Audit logging middleware
export const auditLogger = (action: AuditAction, resource?: string) => {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    // Store original res.json method
    const originalJson = res.json;
    
    // Override res.json to capture response data
    res.json = function(data: unknown) {
      // Log the audit entry after response is sent
      setTimeout(async () => {
        await logAuditEntry(req, res, action, resource, data);
      }, 0);
      
      // Call original json method
      return originalJson.call(this, data);
    };
    
    next();
  };
};

// Function to log audit entries
async function logAuditEntry(
  req: AuthenticatedRequest, 
  res: Response, 
  action: AuditAction, 
  resource?: string, 
  responseData?: unknown
): Promise<void> {
  try {
    const userId = req.user?.id;
    const userEmail = req.user?.email;
    
    // Prepare audit data
    const auditData = {
      user_id: userId || null,
      action,
      resource: resource || `${req.method} ${req.route?.path || req.path}`,
      method: req.method,
      path: req.path,
      ip_address: getClientIP(req),
      user_agent: req.get('User-Agent') || null,
      status_code: res.statusCode,
      success: res.statusCode >= 200 && res.statusCode < 300,
      metadata: {
        email: userEmail,
        timestamp: new Date().toISOString(),
        request_id: req.id || generateRequestId(),
        response_summary: getResponseSummary(responseData),
        request_params: sanitizeRequestParams(req),
        session_id: req.session?.id || null
      }
    };
    
    // Insert audit log into database
    const { error } = await supabase
      .from('audit_logs')
      .insert(auditData);
    
    if (error) {
      console.error('Failed to log audit entry:', error);
      // Don't throw error to avoid breaking the main flow
    }
  } catch (error) {
    console.error('Audit logging error:', error);
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

// Helper function to get response summary
function getResponseSummary(responseData: unknown): string {
  if (!responseData) return 'No response data';
  
  try {
    // If response is too large, truncate it
    const responseStr = JSON.stringify(responseData);
    if (responseStr.length > 500) {
      return responseStr.substring(0, 500) + '...';
    }
    return responseStr;
  } catch {
    return 'Response data not serializable';
  }
}

// Helper function to sanitize request parameters
function sanitizeRequestParams(req: Request): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {};
  
  // Include query parameters (excluding sensitive ones)
  if (req.query) {
    const queryCopy = { ...req.query } as Record<string, unknown>;
    delete queryCopy.password;
    delete queryCopy.token;
    delete queryCopy.api_key;
    sanitized.query = queryCopy;
  }
  
  // Include limited body parameters for POST/PUT requests
  if (req.body && ['POST', 'PUT', 'PATCH'].includes(req.method)) {
    sanitized.body_keys = Object.keys(req.body);
    
    // For specific endpoints, include more details
    if (req.path.includes('/analysis')) {
      sanitized.body = {
        has_file: !!req.file,
        file_type: req.file?.mimetype,
        file_size: req.file?.size
      };
    }
    
    // Exclude sensitive body content
    delete req.body.password;
    delete req.body.token;
    delete req.body.api_key;
    delete req.body.secret;
  }
  
  return sanitized;
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
