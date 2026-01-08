import { AxiosRequestConfig } from 'axios';

export interface PythonApiError {
  error: string;
  code: number;
  details?: Record<string, any>;
}

export interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  maxRetryDelay: number;
  retryBackoffFactor: number;
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  timestamp: string;
  dependencies: Record<string, string>;
}

export interface AnalysisResult {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    code: number;
    details?: Record<string, any>;
  };
  meta?: Record<string, any>;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}
