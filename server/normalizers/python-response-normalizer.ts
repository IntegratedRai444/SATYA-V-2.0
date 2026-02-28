/**
 * 🧠 SINGLE SOURCE OF TRUTH NORMALIZER
 * 
 * THE ONLY normalization layer allowed in SatyaAI
 * All Python responses MUST pass through here
 * NO MORE DUPLICATE NORMALIZERS
 */

import { CanonicalPythonResponse, CanonicalNodeResponse, CanonicalError, ContractGuard } from '../types/canonical-contracts';

/**
 * Converts Python raw response to canonical Node response
 * This is THE ONLY transformation layer allowed
 */
export function normalizePythonResponse(pythonResponse: unknown): CanonicalNodeResponse {
  // Contract guard - fail fast if format is wrong
  ContractGuard.assertValidPythonResponse(pythonResponse);
  
  const response = pythonResponse as CanonicalPythonResponse;
  
  // Handle error responses
  if (!response.success || response.error) {
    return {
      success: false,
      error: {
        code: 'PYTHON_ANALYSIS_ERROR',
        message: response.error || 'Analysis failed',
        details: response
      }
    };
  }
  
  // Convert Python format to canonical Node format
  const isDeepfake = response.label === 'Deepfake';
  const confidence = response.fake_score;
  
  return {
    success: true,
    data: {
      id: `analysis_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      type: response.media_type as 'image' | 'video' | 'audio',
      filename: `analysis_${response.media_type}_${Date.now()}.${getFileExtension(response.media_type)}`,
      result: {
        confidence,
        is_deepfake: isDeepfake,
        model_name: (response.metadata?.model_name as string) || 'unified_detector',
        model_version: (response.metadata?.model_version as string) || '1.0',
        summary: {
          original_label: response.label,
          processing_time_ms: response.processing_time,
          metadata: response.metadata
        }
      },
      timestamp: new Date().toISOString(),
      isDeepfake,
      confidence,
      status: 'completed' as const
    }
  };
}

/**
 * Converts Python errors to canonical error format
 * This replaces ALL duplicate error normalizers
 */
export function normalizePythonError(error: unknown): CanonicalError {
  // Handle FastAPI errors
  if (error && typeof error === 'object') {
    const err = error as Record<string, unknown>;
    
    // FastAPI validation errors
    if (err.detail && typeof err.detail === 'string') {
      return {
        code: 'PYTHON_VALIDATION_ERROR',
        message: err.detail,
        details: error
      };
    }
    
    // Structured errors
    if (err.code && typeof err.code === 'string' && err.message && typeof err.message === 'string') {
      return {
        code: err.code as string,
        message: err.message as string,
        details: err.details || error
      };
    }
    
    // HTTP response errors
    if (err.response && typeof err.response === 'object') {
      const response = err.response as Record<string, unknown>;
      const data = response.data as Record<string, unknown>;
      
      if (data && typeof data === 'object') {
        return {
          code: (data.code as string) || 'PYTHON_HTTP_ERROR',
          message: (data.detail as string) || (data.message as string) || 'Python service error',
          details: data
        };
      }
    }
  }
  
  // Network/system errors
  if (error instanceof Error) {
    const code = (error as { code?: string }).code;
    if (code === 'ECONNREFUSED' || code === 'ETIMEDOUT') {
      return {
        code: 'PYTHON_UNAVAILABLE',
        message: 'AI service temporarily unavailable',
        details: { originalCode: code }
      };
    }
    
    return {
      code: 'PYTHON_RUNTIME_ERROR',
      message: error.message,
      details: { stack: error.stack }
    };
  }
  
  // Fallback
  return {
    code: 'PYTHON_UNKNOWN_ERROR',
    message: 'Unknown Python error occurred',
    details: error
  };
}

/**
 * Validates that a response conforms to canonical format
 * Used as a safety rail before sending to frontend
 */
export function validateCanonicalResponse(response: unknown): CanonicalNodeResponse {
  ContractGuard.assertValidNodeResponse(response);
  return response as CanonicalNodeResponse;
}

/**
 * Creates a canonical error response
 * Standardizes all error responses
 */
export function createCanonicalErrorResponse(code: string, message: string, details?: unknown): CanonicalNodeResponse {
  return {
    success: false,
    error: {
      code,
      message,
      details
    }
  };
}

// Helper function to get file extension
function getFileExtension(mediaType: string): string {
  switch (mediaType) {
    case 'image': return 'jpg';
    case 'video': return 'mp4';
    case 'audio': return 'mp3';
    default: return 'bin';
  }
}
