/**
 * API Type Definitions
 * Comprehensive type definitions for all API requests and responses
 */

// ============================================================================
// Base Types
// ============================================================================

export type ApiStatus = 'success' | 'error' | 'pending';
export type MediaType = 'image' | 'video' | 'audio' | 'multimodal' | 'webcam' | 'batch';
export type AnalysisStatus = 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
export type AuthenticityResult = 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';

// ============================================================================
// Error Types
// ============================================================================

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  statusCode: number;
  timestamp: string;
  path?: string;
}

export interface ValidationError {
  field: string;
  message: string;
  value?: unknown;
}

export interface ErrorResponse {
  success: false;
  error: string;
  errors?: ValidationError[];
  statusCode: number;
  timestamp: string;
}

// ============================================================================
// Generic Response Types
// ============================================================================

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  statusCode?: number;
  timestamp?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasMore: boolean;
    hasPrevious: boolean;
  };
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'user' | 'admin' | 'moderator';
  fullName?: string;
  avatar?: string;
  createdAt: string;
  lastLogin?: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  token?: string;
  user?: User;
  errors?: string[];
}

export interface LoginRequest {
  username: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  fullName?: string;
}

export interface SessionValidation {
  valid: boolean;
  user?: User;
  expiresAt?: string;
}

// ============================================================================
// Analysis Types
// ============================================================================

export interface AnalysisMetrics {
  processingTime: number;
  facesDetected?: number;
  framesAnalyzed?: number;
  audioSegments?: number;
  modelVersion: string;
  confidence: number;
}

export interface AnalysisDetails {
  modelVersion: string;
  analysisMethod: string;
  confidenceBreakdown?: Record<string, number>;
  technicalDetails?: Record<string, unknown>;
  keyFindings: string[];
}

export interface FileInfo {
  originalName: string;
  size: number;
  mimeType: string;
  uploadedAt?: string;
  hash?: string;
}

export interface AnalysisResult {
  id?: string;
  reportCode?: string; // Changed from caseId/jobId to match backend
  type: MediaType;
  status: AnalysisStatus;
  authenticity?: AuthenticityResult;
  confidence?: number;
  analysisDate?: string;
  metrics?: AnalysisMetrics;
  details?: AnalysisDetails;
  fileInfo?: FileInfo;
  error?: ApiError;
  async?: boolean;
  estimatedTime?: number;
}

export interface AnalysisResponse {
  success: boolean;
  message: string;
  result?: AnalysisResult;
  reportCode?: string; // Changed from jobId to match backend
  async?: boolean;
  estimatedTime?: number;
  error?: string;
}

export interface AnalysisOptions {
  sensitivity?: 'low' | 'medium' | 'high';
  includeDetails?: boolean;
  async?: boolean;
  priority?: 'low' | 'normal' | 'high';
}

// ============================================================================
// Dashboard Types
// ============================================================================

export interface ScansByType {
  image: number;
  video: number;
  audio: number;
}

export interface RecentActivity {
  last7Days: number;
  last30Days: number;
  thisMonth: number;
  today?: number;
}

export interface ConfidenceDistribution {
  high: number;    // > 80%
  medium: number;  // 50-80%
  low: number;     // < 50%
}

export interface DashboardStats {
  totalScans: number;
  authenticScans: number;
  manipulatedScans: number;
  uncertainScans: number;
  averageConfidence: number;
  scansByType: ScansByType;
  recentActivity: RecentActivity;
  confidenceDistribution: ConfidenceDistribution;
  topFindings: string[];
}

export interface SystemStats {
  uptime: number;
  cpuUsage: number;
  memoryUsage: number;
  activeUsers: number;
  queuedJobs: number;
}

// ============================================================================
// History Types
// ============================================================================

export interface HistoryItem {
  id: number;
  caseId: string;
  type: MediaType;
  fileName: string;
  fileSize: number;
  authenticity: AuthenticityResult;
  confidence: number;
  analysisDate: string;
  status: AnalysisStatus;
  thumbnailUrl?: string;
}

export interface HistoryFilters {
  type?: MediaType;
  result?: AuthenticityResult;
  dateFrom?: string;
  dateTo?: string;
  minConfidence?: number;
  maxConfidence?: number;
}

export interface HistoryQuery extends HistoryFilters {
  limit?: number;
  offset?: number;
  sortBy?: 'date' | 'confidence' | 'type';
  sortOrder?: 'asc' | 'desc';
}

// ============================================================================
// Upload Types
// ============================================================================

export interface UploadProgress {
  fileId: string;
  fileName: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  result?: AnalysisResult;
}

export interface UploadResponse {
  success: boolean;
  data?: {
    url: string;
    fileId: string;
    fileName: string;
  };
  error?: string;
}

// ============================================================================
// Batch Processing Types
// ============================================================================

export interface BatchItem {
  fileId: string;
  file: File;
  status: AnalysisStatus;
  progress: number;
  result?: AnalysisResult;
  error?: string;
}

export interface BatchAnalysisRequest {
  files: File[];
  type: MediaType;
  options?: AnalysisOptions;
}

export interface BatchAnalysisResponse {
  success: boolean;
  batchId: string;
  totalFiles: number;
  results: AnalysisResult[];
  errors?: ApiError[];
}

// ============================================================================
// Health Check Types
// ============================================================================

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  environment: string;
  uptime: number;
  services?: {
    database: 'up' | 'down';
    python: 'up' | 'down';
    redis?: 'up' | 'down';
  };
}

// ============================================================================
// Notification Types
// ============================================================================

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
  actionLabel?: string;
}

// ============================================================================
// Analytics Types
// ============================================================================

export interface AnalyticsData {
  period: 'day' | 'week' | 'month' | 'year';
  scans: {
    total: number;
    authentic: number;
    manipulated: number;
    uncertain: number;
  };
  trends: {
    date: string;
    count: number;
    authentic: number;
    manipulated: number;
  }[];
  topModels: {
    name: string;
    usage: number;
  }[];
}

// ============================================================================
// Settings Types
// ============================================================================

export interface UserSettings {
  notifications: {
    email: boolean;
    push: boolean;
    analysisComplete: boolean;
    weeklyReport: boolean;
  };
  privacy: {
    shareAnalytics: boolean;
    publicProfile: boolean;
  };
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    language: string;
    timezone: string;
  };
}

export interface UpdateSettingsRequest {
  settings: Partial<UserSettings>;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    'message' in error &&
    'statusCode' in error
  );
}

export function isErrorResponse(response: unknown): response is ErrorResponse {
  return (
    typeof response === 'object' &&
    response !== null &&
    'success' in response &&
    (response as ErrorResponse).success === false
  );
}

export function isAnalysisResult(data: unknown): data is AnalysisResult {
  return (
    typeof data === 'object' &&
    data !== null &&
    'type' in data &&
    'status' in data
  );
}
