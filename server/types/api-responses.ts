// Standardized API Response Types for SatyaAI

export interface BaseResponse {
  success: boolean;
  message?: string;
  timestamp?: string;
}

export interface SuccessResponse<T = any> extends BaseResponse {
  success: true;
  data?: T;
}

export interface ErrorResponse extends BaseResponse {
  success: false;
  error?: {
    code: string;
    message: string;
    details?: string;
    suggestions?: string[];
  };
}

export interface AnalysisResponse extends BaseResponse {
  data?: {
    analysisId: string;
    results?: DetectionResults;
    confidence?: number;
    processingTime?: number;
    metadata?: FileMetadata;
  };
  progress?: {
    percentage: number;
    stage: string;
    estimatedTimeRemaining?: number;
  };
}

export interface DetectionResults {
  authenticity: 'real' | 'fake' | 'uncertain';
  confidence: number;
  score: number;
  details: {
    method: string;
    modelUsed: string;
    processingTime: number;
    keyFindings: string[];
  };
  metadata?: {
    fileSize: number;
    dimensions?: { width: number; height: number };
    duration?: number;
    format: string;
  };
}

export interface FileMetadata {
  originalName: string;
  size: number;
  mimeType: string;
  uploadedAt: string;
}

export interface ProgressUpdate {
  analysisId: string;
  type: 'progress' | 'complete' | 'error';
  data: {
    percentage?: number;
    stage?: string;
    results?: DetectionResults;
    error?: ErrorInfo;
  };
}

export interface ErrorInfo {
  code: string;
  message: string;
  userMessage: string;
  suggestions: string[];
  retryable: boolean;
  retryAfter?: number;
}

export enum ErrorCategory {
  USER_INPUT = 'user_input',
  SYSTEM_ERROR = 'system_error',
  NETWORK_ERROR = 'network_error',
  AI_SERVICE_ERROR = 'ai_service_error',
  AUTHENTICATION_ERROR = 'auth_error'
}

// Helper functions for creating standardized responses
export function createSuccessResponse<T>(data?: T, message?: string): SuccessResponse<T> {
  return {
    success: true,
    data,
    message,
    timestamp: new Date().toISOString()
  };
}

export function createErrorResponse(
  code: string,
  message: string,
  details?: string,
  suggestions?: string[]
): ErrorResponse {
  return {
    success: false,
    message: 'Request failed',
    timestamp: new Date().toISOString(),
    error: {
      code,
      message,
      details,
      suggestions
    }
  };
}

export function createAnalysisResponse(
  analysisId: string,
  results?: DetectionResults,
  progress?: { percentage: number; stage: string; estimatedTimeRemaining?: number }
): AnalysisResponse {
  return {
    success: true,
    timestamp: new Date().toISOString(),
    data: results ? {
      analysisId,
      results,
      confidence: results.confidence,
      processingTime: results.details.processingTime,
      metadata: results.metadata ? {
        originalName: '',
        size: results.metadata.fileSize,
        mimeType: results.metadata.format,
        uploadedAt: new Date().toISOString()
      } : undefined
    } : undefined,
    progress
  };
}

// Common error codes
export const ERROR_CODES = {
  // User Input Errors
  NO_FILE: 'NO_FILE',
  INVALID_FILE_TYPE: 'INVALID_FILE_TYPE',
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  INVALID_IMAGE_DATA: 'INVALID_IMAGE_DATA',
  MISSING_ID: 'MISSING_ID',
  
  // System Errors
  ANALYSIS_ERROR: 'ANALYSIS_ERROR',
  PROCESSING_ERROR: 'PROCESSING_ERROR',
  STATUS_ERROR: 'STATUS_ERROR',
  RESULTS_ERROR: 'RESULTS_ERROR',
  
  // Authentication Errors
  ACCESS_DENIED: 'ACCESS_DENIED',
  INVALID_TOKEN: 'INVALID_TOKEN',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',
  
  // Resource Errors
  NOT_FOUND: 'NOT_FOUND',
  NOT_COMPLETED: 'NOT_COMPLETED',
  NO_RESULTS: 'NO_RESULTS',
  
  // Service Errors
  AI_SERVICE_UNAVAILABLE: 'AI_SERVICE_UNAVAILABLE',
  DATABASE_ERROR: 'DATABASE_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR'
} as const;

// Common error messages with suggestions
export const ERROR_MESSAGES = {
  [ERROR_CODES.NO_FILE]: {
    message: 'No file was provided for analysis',
    suggestions: ['Please select a file to upload', 'Ensure the file is properly attached to your request']
  },
  [ERROR_CODES.INVALID_FILE_TYPE]: {
    message: 'The uploaded file type is not supported',
    suggestions: ['Please upload an image (JPEG, PNG, GIF, WebP)', 'For video: MP4, AVI, MOV, WebM', 'For audio: MP3, WAV, FLAC']
  },
  [ERROR_CODES.FILE_TOO_LARGE]: {
    message: 'The uploaded file exceeds the size limit',
    suggestions: ['Images: max 10MB', 'Videos: max 100MB', 'Audio: max 50MB', 'Try compressing your file']
  },
  [ERROR_CODES.AI_SERVICE_UNAVAILABLE]: {
    message: 'The AI analysis service is temporarily unavailable',
    suggestions: ['Please try again in a few moments', 'Check system status for updates', 'Contact support if the issue persists']
  },
  [ERROR_CODES.ACCESS_DENIED]: {
    message: 'You do not have permission to access this resource',
    suggestions: ['Please log in to your account', 'Ensure you are accessing your own analysis results']
  },
  [ERROR_CODES.NOT_FOUND]: {
    message: 'The requested analysis was not found',
    suggestions: ['Check the analysis ID is correct', 'The analysis may have been deleted', 'Try refreshing your analysis history']
  }
} as const;