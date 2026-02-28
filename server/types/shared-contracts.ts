/**
 * Shared Contracts - Single Source of Truth for API Contracts
 * 
 * This file eliminates runtime drift by providing strict types
 * across frontend, Node backend, and Python services.
 */

// UNIVERSAL RESPONSE FORMAT
export interface UniversalAnalysisResponse {
  success: boolean;
  media_type: 'image' | 'video' | 'audio' | 'multimodal';
  prediction: 'real' | 'fake';
  confidence: number;
  reasoning?: string | null;
  metadata: Record<string, unknown>;
  processing_time_ms?: number;
}

export interface UniversalErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export type UniversalResponse = UniversalAnalysisResponse | UniversalErrorResponse;

// JOB MANAGEMENT TYPES
export interface JobStartResponse {
  success: true;
  jobId: string;
  status: 'processing';
}

export interface JobProgress {
  jobId: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string;
  timestamp: number;
}

export interface JobResult {
  success: boolean;
  media_type: string;
  prediction: 'real' | 'fake';
  confidence: number;
  reasoning?: string;
  metadata: Record<string, unknown>;
  processing_time_ms?: number;
}

// WEBSOCKET MESSAGE TYPES
export interface WebSocketMessageBase {
  type: string;
  timestamp: number;
  requestId?: string;
}

export interface JobProgressMessage extends WebSocketMessageBase {
  type: 'JOB_PROGRESS';
  jobId: string;
  data: {
    stage: string;
    progress: number;
  };
}

export interface JobCompletedMessage extends WebSocketMessageBase {
  type: 'JOB_COMPLETED';
  jobId: string;
  data: {
    status: 'completed';
    confidence: number;
    is_deepfake: boolean;
    model_name: string;
    model_version: string;
  };
}

export interface JobFailedMessage extends WebSocketMessageBase {
  type: 'JOB_FAILED';
  jobId: string;
  data: {
    error: string;
  };
}

export type WebSocketMessage = JobProgressMessage | JobCompletedMessage | JobFailedMessage;

// REQUEST CONTEXT TYPES
export interface UserContext {
  id: string;
  email: string;
  role?: string;
}

export interface RequestHeaders {
  Authorization?: string;
  'X-Request-Id'?: string;
  'X-User-ID'?: string;
  'Content-Type'?: string;
}

export interface RequestContext {
  userContext?: UserContext;
  headers?: RequestHeaders;
  requestId?: string;
}

// ERROR TYPES
export interface ApiError {
  code: string | number;
  message: string;
  details?: Record<string, unknown>;
  isTimeout?: boolean;
}

export interface ValidationError {
  field: string;
  message: string;
  value?: unknown;
}

// ANALYSIS REQUEST TYPES
export interface AnalysisRequest {
  file: Buffer | File;
  job_id: string;
  user_id: string;
  request_id?: string;
}

export interface AnalysisMetadata {
  originalName: string;
  mimeType: string;
  size: number;
  requestId?: string;
}

// STRICT TYPE GUARDS
export function isUniversalResponse(obj: unknown): obj is UniversalResponse {
  return typeof obj === 'object' && obj !== null && 'success' in obj;
}

export function isJobProgressMessage(obj: unknown): obj is JobProgressMessage {
  return typeof obj === 'object' && obj !== null && 
         'type' in obj && obj.type === 'JOB_PROGRESS';
}

export function isJobCompletedMessage(obj: unknown): obj is JobCompletedMessage {
  return typeof obj === 'object' && obj !== null && 
         'type' in obj && obj.type === 'JOB_COMPLETED';
}

export function isJobFailedMessage(obj: unknown): obj is JobFailedMessage {
  return typeof obj === 'object' && obj !== null && 
         'type' in obj && obj.type === 'JOB_FAILED';
}

// VALIDATION HELPERS
export function validateAnalysisResponse(response: unknown): UniversalAnalysisResponse {
  if (!isUniversalResponse(response) || !response.success) {
    throw new Error('Invalid analysis response format');
  }
  
  const successResponse = response as UniversalAnalysisResponse;
  
  if (typeof successResponse.confidence !== 'number' || 
      successResponse.confidence < 0 || 
      successResponse.confidence > 1) {
    throw new Error(`Invalid confidence value: ${successResponse.confidence}`);
  }
  
  if (!['real', 'fake'].includes(successResponse.prediction)) {
    throw new Error(`Invalid prediction value: ${successResponse.prediction}`);
  }
  
  if (!['image', 'video', 'audio', 'multimodal'].includes(successResponse.media_type)) {
    throw new Error(`Invalid media_type: ${successResponse.media_type}`);
  }
  
  return successResponse;
}

export function validateErrorResponse(response: unknown): UniversalErrorResponse {
  if (!isUniversalResponse(response) || response.success) {
    throw new Error('Invalid error response format');
  }
  
  const errorResponse = response as UniversalErrorResponse;
  
  if (!errorResponse.error || typeof errorResponse.error.code !== 'string' || typeof errorResponse.error.message !== 'string') {
    throw new Error('Invalid error format');
  }
  
  return errorResponse;
}
