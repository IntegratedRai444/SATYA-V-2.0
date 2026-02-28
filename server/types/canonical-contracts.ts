/**
 * 🧠 SATYAAI CANONICAL CONTRACTS
 * 
 * Single source of truth for all API contracts
 * NO MORE DRIFT - THIS IS THE LAW
 */

// ===== CANONICAL REQUEST CONTRACTS =====

export interface CanonicalUploadRequest {
  file: Blob;
  media_type: 'image' | 'video' | 'audio';
}

export interface CanonicalPythonRequest {
  file: Blob;
  job_id: string;
  user_id: string;
  media_type: 'image' | 'video' | 'audio';
}

// ===== CANONICAL RESPONSE CONTRACTS =====

export interface CanonicalPythonResponse {
  success: boolean;
  media_type: string;
  fake_score: number; // 0.0-1.0
  label: 'Deepfake' | 'Authentic' | 'Unknown';
  processing_time: number; // milliseconds
  metadata: {
    model_name?: string;
    model_version?: string;
    [key: string]: unknown;
  };
  error?: string;
}

export interface CanonicalNodeResponse {
  success: boolean;
  data?: {
    id: string;
    type: 'image' | 'video' | 'audio';
    filename: string;
    result: {
      confidence: number; // 0.0-1.0
      is_deepfake: boolean;
      model_name?: string;
      model_version?: string;
      summary?: Record<string, unknown>;
    };
    timestamp: string;
    isDeepfake: boolean;
    confidence: number;
    status: 'completed' | 'processing' | 'failed';
  };
  error?: CanonicalError;
}

// ===== CANONICAL ERROR CONTRACT =====

export interface CanonicalError {
  code: string;
  message: string;
  details?: unknown;
}

export interface CanonicalErrorResponse {
  success: false;
  error: CanonicalError;
}

// ===== CANONICAL HEADER CONTRACTS =====

export interface CanonicalHeaders {
  Authorization: string;
  'X-Request-ID': string; // UPPERCASE ID - STANDARDIZED
  'X-User-ID': string;
  'X-User-Email': string;
}

// ===== CANONICAL TIMEOUT CONTRACTS =====

export const CANONICAL_TIMEOUTS = {
  FRONTEND_ANALYSIS: 10 * 60 * 1000, // 10 minutes
  NODE_BRIDGE: 15 * 60 * 1000, // 15 minutes
  PYTHON_ML: 15 * 60 * 1000, // 15 minutes
  FRONTEND_UPLOAD: 2 * 60 * 1000, // 2 minutes
  FRONTEND_GET: 30 * 1000, // 30 seconds
} as const;

// ===== CANONICAL ROUTE CONTRACTS =====

export const CANONICAL_ROUTES = {
  // Frontend → Node
  FRONTEND_UPLOAD: '/api/v2/analysis/unified/{type}',
  
  // Node → Python  
  PYTHON_ANALYZE: '/api/v2/analysis/analyze/{media_type}',
  
  // Standardized endpoints
  HISTORY: '/api/v2/analysis/history',
  RESULTS: '/api/v2/analysis/results/{id}',
  HEALTH: '/api/v2/health',
} as const;

/**
 * 🚨 CONTRACT VIOLATION DETECTION
 * 
 * These helpers detect if contracts are being violated
 */
export class ContractGuard {
  static validatePythonResponse(response: unknown): response is CanonicalPythonResponse {
    if (!response || typeof response !== 'object') return false;
    
    const r = response as Record<string, unknown>;
    return (
      typeof r.success === 'boolean' &&
      typeof r.media_type === 'string' &&
      typeof r.fake_score === 'number' &&
      typeof r.label === 'string' &&
      typeof r.processing_time === 'number'
    );
  }
  
  static validateNodeResponse(response: unknown): response is CanonicalNodeResponse {
    if (!response || typeof response !== 'object') return false;
    
    const r = response as Record<string, unknown>;
    return (
      typeof r.success === 'boolean' &&
      (r.success === false || (r.data && typeof r.data === 'object'))
    );
  }
  
  static validateError(error: unknown): error is CanonicalError {
    if (!error || typeof error !== 'object') return false;
    
    const e = error as Record<string, unknown>;
    return (
      typeof e.code === 'string' &&
      typeof e.message === 'string'
    );
  }
  
  static assertValidPythonResponse(response: unknown): asserts response is CanonicalPythonResponse {
    if (!this.validatePythonResponse(response)) {
      throw new Error('Contract violation: Invalid Python response format');
    }
  }
  
  static assertValidNodeResponse(response: unknown): asserts response is CanonicalNodeResponse {
    if (!response || typeof response !== 'object') {
      throw new Error('Contract violation: Response must be an object');
    }
    
    const r = response as Record<string, unknown>;
    if (typeof r.success !== 'boolean') {
      throw new Error('Contract violation: Invalid Node response format');
    }
  }
  
  static assertValidError(error: unknown): asserts error is CanonicalError {
    if (!this.validateError(error)) {
      throw new Error('Contract violation: Invalid error format');
    }
  }
}
