/**
 * SatyaAI API Client
 * Handles all communication with the backend API
 */

import axios, { type AxiosInstance } from 'axios';
import logger from './logger';
import { handleError, classifyError } from './errorHandler';
import { createErrorInterceptor } from '../utils/apiErrorHandler';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 300000; // 5 minutes for analysis endpoints

// Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  statusCode?: number;
  [key: string]: any; // Allow additional properties
}

export interface AuthResponse {
  success: boolean;
  message: string;
  token?: string;
  user?: {
    id: number;
    username: string;
    email?: string;
    fullName?: string;
    role: string;
  };
  errors?: string[];
}

export interface AnalysisResult {
  success: boolean;
  message: string;
  result?: {
    authenticity: 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';
    confidence: number;
    analysisDate: string;
    caseId: string;
    keyFindings: string[];
    metrics?: {
      processingTime: number;
      facesDetected?: number;
      framesAnalyzed?: number;
      audioSegments?: number;
    };
    details?: {
      modelVersion: string;
      analysisMethod: string;
      confidenceBreakdown?: Record<string, number>;
      technicalDetails?: Record<string, any>;
    };
    fileInfo?: {
      originalName: string;
      size: number;
      mimeType: string;
    };
    modalityResults?: {
      [key: string]: {
        authenticity: 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';
        confidence: number;
      };
    };
    fusionAnalysis?: {
      aggregatedScore: number;
      consistencyScore: number;
      confidenceLevel: 'low' | 'medium' | 'high';
      conflictsDetected: string[];
    };
  };
  jobId?: string;
  async?: boolean;
  estimatedTime?: number;
  error?: string;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
  environment: string;
  uptime: number;
}

export interface DashboardStats {
  totalScans: number;
  authenticScans: number;
  manipulatedScans: number;
  uncertainScans: number;
  averageConfidence: number;
  scansByType: {
    image: number;
    video: number;
    audio: number;
  };
  recentActivity: {
    last7Days: number;
    last30Days: number;
    thisMonth: number;
  };
  confidenceDistribution: {
    high: number;
    medium: number;
    low: number;
  };
  topFindings: string[];
}

// Create axios instance
class ApiClient {
  public client: AxiosInstance;
  public authToken: string | null = null;
  private connectionStatus: {
    backend: boolean;
    lastChecked: Date | null;
    retryCount: number;
  } = {
    backend: false,
    lastChecked: null,
    retryCount: 0
  } as const;

  constructor() {
    // Initialize Axios instance with proper TypeScript types
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      withCredentials: true, // Enable sending cookies with cross-origin requests
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      // Handle CSRF protection
      xsrfCookieName: 'XSRF-TOKEN',
      xsrfHeaderName: 'X-XSRF-TOKEN'
    } as any); // Type assertion to handle custom properties

    // Load auth token from localStorage (use same key as auth service)
    this.authToken = localStorage.getItem('satyaai_auth_token');
    if (this.authToken) {
      // Check if token has expiry set, if not, set it with default
      const expiry = localStorage.getItem('satyaai_token_expiry');
      if (!expiry) {
        // Set default expiry for existing tokens without expiry
        this.setAuthToken(this.authToken, 24 * 60 * 60 * 1000);
      } else {
        // Just set the token without updating expiry
        this.client.defaults.headers.common['Authorization'] = `Bearer ${this.authToken}`;
      }
    }

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }

        // Enable credentials for CORS (cookies, auth headers)
        config.withCredentials = true;

        // Enhanced logging in development
        logger.debug(`API Request: ${config.method?.toUpperCase()} ${config.url}`, {
          url: config.url,
          method: config.method,
          timeout: config.timeout,
          baseURL: config.baseURL,
          withCredentials: config.withCredentials,
          hasData: !!config.data
        });

        return config;
      },
      (error) => {
        logger.error('Request interceptor error', error, {
          hasResponse: !!error.response,
          hasRequest: !!error.request,
          status: error.response?.status
        });
        return Promise.reject(error);
      }
    );

    // Response interceptor with token refresh
    this.client.interceptors.response.use(
      (response) => {
        // Transform response format for consistency
        // Backend returns 'result' for analysis endpoints, transform to 'data' for consistency
        if (response.data && response.data.result && !response.data.data) {
          response.data.data = response.data.result;
        }

        // Enhanced logging in development
        logger.debug(`API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, {
          status: response.status,
          statusText: response.statusText,
          hasData: !!response.data
        });
        return response;
      },
      async (error) => {
        // Enhanced error logging
        const classified = classifyError(error);
        logger.error(`API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url}`, error, {
          type: classified.type,
          status: error.response?.status,
          code: error.code,
          url: error.config?.url
        });

        const originalRequest = error.config;

        // Handle auth errors with token refresh attempt
        if (error.response?.status === 401 && !originalRequest._retry && originalRequest.url !== '/api/auth/refresh' && originalRequest.url !== '/api/auth/login') {
          originalRequest._retry = true;

          try {
            logger.info('401 error, attempting to refresh token');
            // Try to refresh the token using a direct axios call to avoid interceptor loop
            const refreshResponse = await axios.post(
              `${API_BASE_URL}/api/auth/refresh`,
              {},
              {
                headers: {
                  'Authorization': `Bearer ${this.authToken}`,
                  'Content-Type': 'application/json'
                },
                withCredentials: true
              }
            );

            if (refreshResponse.data.success && refreshResponse.data.token) {
              // Update token and retry original request
              logger.info('Token refreshed successfully');
              this.setAuthToken(refreshResponse.data.token);
              originalRequest.headers.Authorization = `Bearer ${refreshResponse.data.token}`;
              return this.client(originalRequest);
            } else {
              // Token refresh failed, clear auth only if not a network error
              logger.info('Token refresh failed, clearing auth');
              this.clearAuth();
            }
          } catch (refreshError: any) {
            // Only clear auth if it's an actual auth error, not a network error
            if (refreshError.response?.status === 401) {
              logger.error('Token refresh returned 401, clearing auth', refreshError as Error);
              this.clearAuth();
            } else {
              logger.error('Token refresh network error, keeping auth', refreshError as Error);
              // Don't clear auth on network errors, just reject the request
            }
          }
        }

        // Handle CORS errors
        if (error.message?.includes('Network Error') && !error.response) {
          error.isCorsError = true;
        }

        // Handle rate limiting
        if (error.response?.status === 429) {
          logger.warn('Rate limit exceeded');
        }

        // Use centralized error handler for consistent error handling
        // This will log and show notifications automatically
        createErrorInterceptor('API Request')(error);

        return Promise.reject(error);
      }
    );
  }

  // Auth methods
  setAuthToken(token: string, expiresIn: number = 24 * 60 * 60 * 1000) {
    this.authToken = token;
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    localStorage.setItem('satyaai_auth_token', token);
    // Set expiry time
    const expiryTime = Date.now() + expiresIn;
    localStorage.setItem('satyaai_token_expiry', expiryTime.toString());
  }

  clearAuth() {
    this.authToken = null;
    delete this.client.defaults.headers.common['Authorization'];
    localStorage.removeItem('satyaai_auth_token');
    localStorage.removeItem('satyaai_token_expiry');
    // Dispatch custom event to notify AuthContext
    window.dispatchEvent(new CustomEvent('auth-cleared'));
  }

  // Authentication endpoints
  async login(username: string, password: string): Promise<AuthResponse> {
    try {
      logger.debug('Making login request');
      const response = await this.client.post('/api/auth/login', {
        username,
        password,
      });

      logger.info('Login successful');

      if (response.data.success && response.data.token) {
        this.setAuthToken(response.data.token);
      }

      return response.data;
    } catch (error: any) {
      logger.error('Login request failed', error);
      await handleError(error, { showToast: false });
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Login failed',
        errors: errorData?.errors
      };
    }
  }

  async register(username: string, email: string, password: string, fullName?: string): Promise<AuthResponse> {
    try {
      const response = await this.client.post('/api/auth/register', {
        username,
        email,
        password,
        fullName,
      });

      if (response.data.success && response.data.token) {
        this.setAuthToken(response.data.token);
      }

      return response.data;
    } catch (error: any) {
      logger.error('Registration request failed', error);
      await handleError(error, { showToast: false });
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Registration failed',
        errors: errorData?.errors
      };
    }
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/api/auth/logout');
    } catch (error) {
      logger.warn('Logout request failed', { error });
    } finally {
      this.clearAuth();
    }
  }

  async validateSession(): Promise<{ valid: boolean; user?: any }> {
    try {
      const sessionResponse = await this.client.get('/api/auth/session');
      return {
        valid: sessionResponse.data.success,
        user: sessionResponse.data.user
      };
    } catch (error) {
      return { valid: false };
    }
  }

  async refreshToken(): Promise<{ success: boolean; token?: string; user?: any }> {
    try {
      const refreshResponse = await this.client.post('/api/auth/refresh');
      if (refreshResponse.data.success && refreshResponse.data.token) {
        this.setAuthToken(refreshResponse.data.token);
        return {
          success: true,
          token: refreshResponse.data.token,
          user: refreshResponse.data.user
        };
      }
      return { success: false };
    } catch (error) {
      logger.error('Token refresh failed', error as Error);
      return { success: false };
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<ApiResponse> {
    try {
      const response = await this.client.post('/api/auth/change-password', {
        currentPassword,
        newPassword,
      });
      return response.data;
    } catch (error: any) {
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Password change failed',
        error: errorData?.errors?.join(', ') || errorData?.error
      };
    }
  }

  async getProfile(): Promise<ApiResponse> {
    try {
      const response = await this.client.get('/api/auth/profile');
      return response.data;
    } catch (error: any) {
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Failed to get profile',
      };
    }
  }

  async updateProfile(data: { fullName?: string; email?: string }): Promise<ApiResponse> {
    try {
      const response = await this.client.put('/api/auth/profile', data);
      return response.data;
    } catch (error: any) {
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Failed to update profile',
        error: errorData?.errors?.join(', ') || errorData?.error
      };
    }
  }

  async deleteAccount(): Promise<ApiResponse> {
    try {
      const response = await this.client.delete('/api/auth/account');
      return response.data;
    } catch (error: any) {
      const errorData = error.response?.data;
      return {
        success: false,
        message: errorData?.message || 'Failed to delete account',
      };
    }
  }

  // Health and status endpoints
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<any> {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }

  async checkPythonHealth(): Promise<{ healthy: boolean; message?: string }> {
    try {
      const response = await this.client.get('/api/health/python');
      return { healthy: response.data.success, message: response.data.message };
    } catch (error) {
      return { healthy: false, message: 'Python server not responding' };
    }
  }

  // Analysis endpoints
  async analyzeImage(imageFile: File | string, options?: {
    sensitivity?: 'low' | 'medium' | 'high';
    includeDetails?: boolean;
    async?: boolean;
  }): Promise<AnalysisResult> {
    if (typeof imageFile === 'string') {
      // Webcam analysis
      const response = await this.client.post<AnalysisResult>('/api/analysis/webcam', {
        imageData: imageFile,
        options,
      });
      return response.data;
    } else {
      // File upload
      const formData = new FormData();
      formData.append('image', imageFile);
      if (options) {
        formData.append('options', JSON.stringify(options));
      }

      const response = await this.client.post<AnalysisResult>('/api/analysis/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for image analysis
      });
      return response.data;
    }
  }

  async analyzeVideo(videoFile: File, options?: {
    sensitivity?: 'low' | 'medium' | 'high';
    includeDetails?: boolean;
    async?: boolean;
  }): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('video', videoFile);
    if (options) {
      formData.append('options', JSON.stringify(options));
    }

    const response = await this.client.post<AnalysisResult>('/api/analysis/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes for video analysis
    });
    return response.data;
  }

  async analyzeAudio(audioFile: File, options?: {
    sensitivity?: 'low' | 'medium' | 'high';
    includeDetails?: boolean;
    async?: boolean;
  }): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    if (options) {
      formData.append('options', JSON.stringify(options));
    }

    const response = await this.client.post<AnalysisResult>('/api/analysis/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 180000, // 3 minutes for audio analysis
    });
    return response.data;
  }

  async analyzeMultimodal(files: {
    image?: File;
    video?: File;
    audio?: File;
  }, options?: {
    sensitivity?: 'low' | 'medium' | 'high';
    includeDetails?: boolean;
    async?: boolean;
  }): Promise<AnalysisResult> {
    const formData = new FormData();

    if (files.image) formData.append('image', files.image);
    if (files.video) formData.append('video', files.video);
    if (files.audio) formData.append('audio', files.audio);
    if (options) {
      formData.append('options', JSON.stringify(options));
    }

    const response = await this.client.post<AnalysisResult>('/api/analysis/multimodal', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes for multimodal analysis
    });
    return response.data;
  }

  async analyzeWebcam(imageData: string, options?: {
    sensitivity?: 'low' | 'medium' | 'high';
    includeDetails?: boolean;
  }): Promise<AnalysisResult> {
    const response = await this.client.post<AnalysisResult>('/api/analysis/webcam', {
      imageData: imageData,
      options,
    });
    return response.data;
  }

  // Job management for async analysis
  async getAnalysisResult(jobId: string): Promise<ApiResponse> {
    const response = await this.client.get(`/api/analysis/result/${jobId}`);
    return response.data;
  }

  async getAnalysisHistory(params?: {
    limit?: number;
    offset?: number;
    type?: string;
    status?: string;
  }): Promise<ApiResponse> {
    const response = await this.client.get('/api/analysis/history', { params });
    return response.data;
  }

  // Dashboard endpoints
  async getDashboardStats(): Promise<ApiResponse<DashboardStats>> {
    const response = await this.client.get('/api/dashboard/stats');
    return response.data;
  }

  async getUserAnalytics(): Promise<ApiResponse> {
    const response = await this.client.get('/api/dashboard/analytics');
    return response.data;
  }

  async getScans(params?: {
    limit?: number;
    offset?: number;
    type?: string;
    result?: string;
    dateFrom?: string;
    dateTo?: string;
    sortBy?: string;
    sortOrder?: string;
  }): Promise<ApiResponse> {
    const response = await this.client.get('/api/dashboard/scans', { params });
    return response.data;
  }

  async getScan(id: number): Promise<ApiResponse> {
    const response = await this.client.get(`/api/dashboard/scans/${id}`);
    return response.data;
  }

  async getRecentActivity(): Promise<ApiResponse> {
    const response = await this.client.get('/api/dashboard/recent-activity');
    return response.data;
  }

  async getSystemStats(): Promise<ApiResponse> {
    const response = await this.client.get('/api/dashboard/system-stats');
    return response.data;
  }

  // Upload endpoints
  public uploadImage = async (imageFile: File): Promise<ApiResponse<{ url: string }>> => {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await this.client.post<Omit<ApiResponse, 'success'>>(
        '/api/upload/image',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return {
        success: true,
        ...response.data,
        data: response.data.data || { url: '' }
      };
    } catch (error) {
      return this.handleError(error);
    }
  }

  public uploadVideo = async (videoFile: File): Promise<ApiResponse<{ url: string }>> => {
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await this.client.post<Omit<ApiResponse, 'success'>>(
        '/api/upload/video',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return {
        success: true,
        ...response.data,
        data: response.data.data || { url: '' }
      };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async uploadAudio(audioFile: File): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('audio', audioFile);

    const response = await this.client.post('/api/upload/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async deleteFile(filename: string): Promise<ApiResponse> {
    const response = await this.client.delete(`/api/upload/${filename}`);
    return response.data;
  }

  async getFileInfo(filename: string): Promise<ApiResponse> {
    const response = await this.client.get(`/api/upload/info/${filename}`);
    return response.data;
  }

  // Handle API errors consistently
  private handleError = (error: unknown): never => {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        logger.error('API Error Response', error, {
          status: error.response.status,
          statusText: error.response.statusText
        });

        const errorData = error.response.data as Record<string, any>;
        const errorMessage = errorData?.message || error.message || 'An unknown error occurred';
        const errorResponse: ApiResponse = {
          success: false,
          error: errorMessage,
          statusCode: error.response.status,
          ...(errorData || {})
        };

        throw errorResponse as never;
      } else if (error.request) {
        logger.error('No response received from server', error);
        const errorResponse: ApiResponse = {
          success: false,
          error: 'No response from server. Please check your network connection.',
          statusCode: 0
        };
        throw errorResponse as never;
      }
    }

    // Handle non-Axios errors
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    logger.error('API Error', error as Error);
    const errorResponse: ApiResponse = {
      success: false,
      error: errorMessage,
      statusCode: 0
    };
    throw errorResponse as never;
  };

  // Connection health checking methods
  async checkBackendHealth(): Promise<{ connected: boolean; responseTime?: number; error?: string }> {
    try {
      const startTime = Date.now();
      // Use _ to indicate we're intentionally not using the response
      await this.client.get('/health', {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate'
        },
        withCredentials: true
      });
      const endTime = Date.now();
      return {
        connected: true,
        responseTime: endTime - startTime
      };
    } catch (error) {
      return {
        connected: false,
        error: error instanceof Error ? error.message : 'Connection failed'
      };
    }
  }

  async retryConnection(maxRetries: number = 3): Promise<boolean> {
    logger.info(`Retrying connection (attempt ${this.connectionStatus.retryCount + 1}/${maxRetries})`);

    for (let i = 0; i < maxRetries; i++) {
      const health = await this.checkBackendHealth();
      if (health.connected) {
        return true;
      }

      const delay = Math.pow(2, i) * 1000;
      logger.debug(`Waiting ${delay}ms before next retry`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }

    return false;
  }

  getConnectionStatus() {
    return { ...this.connectionStatus };
  }

  // Request method for making API calls
  async request<T = any>(config: {
    method: 'get' | 'post' | 'put' | 'delete' | 'patch';
    url: string;
    data?: any;
    params?: any;
    headers?: Record<string, string>;
    onUploadProgress?: (progressEvent: any) => void;
  }): Promise<{ data: T }> {
    try {
      const response = await this.client.request<T>({
        method: config.method,
        url: config.url,
        data: config.data,
        params: config.params,
        headers: config.headers,
        onUploadProgress: config.onUploadProgress,
      });
      return { data: response.data };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.message || error.message);
      }
      throw error;
    }
  }

  // Utility methods
  isAuthenticated(): boolean {
    return !!this.authToken;
  }

  getAuthToken(): string | null {
    return this.authToken;
  }

  // Get base URL for external use
  getBaseURL(): string {
    return API_BASE_URL;
  }

  // Development configuration diagnostics
  validateConfiguration(): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    if (!API_BASE_URL) {
      issues.push('VITE_API_URL is not set');
    }

    if (API_BASE_URL === 'http://localhost:3000' && import.meta.env.DEV) {
      // This is expected in development
    } else if (API_BASE_URL.includes('localhost') && !import.meta.env.DEV) {
      issues.push('Using localhost URL in production build');
    }

    if (API_TIMEOUT < 5000) {
      issues.push('API timeout is very low (< 5 seconds)');
    }

    return {
      valid: issues.length === 0,
      issues
    };
  }

  // Development diagnostics
  async runDiagnostics(): Promise<void> {
    if (!import.meta.env.DEV) return;

    console.group('🔧 SatyaAI API Client Diagnostics');

    const config = this.validateConfiguration();
    logger.debug('API Configuration', {
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      valid: config.valid,
      issues: config.issues
    });

    logger.debug('Environment', {
      NODE_ENV: import.meta.env.MODE,
      DEV: import.meta.env.DEV
    });

    const health = await this.checkBackendHealth();
    logger.debug('Backend health check', health);

    logger.debug('Authentication status', {
      hasToken: !!this.authToken
    });

    console.groupEnd();
  }
}

// Create a singleton instance
const apiClient = new ApiClient();

// Export the singleton instance as default
export default apiClient;

// Export convenience methods
export const {
  login,
  register,
  logout,
  validateSession,
  changePassword,
  getProfile,
  getHealth,
  getDetailedHealth,
  analyzeImage,
  analyzeVideo,
  analyzeAudio,
  analyzeMultimodal,
  analyzeWebcam,
  getAnalysisResult,
  getAnalysisHistory,
  getDashboardStats,
  getUserAnalytics,
  getScans,
  getScan,
  getRecentActivity,
  getSystemStats,
  uploadImage,
  uploadVideo,
  uploadAudio,
  deleteFile,
  getFileInfo,
  isAuthenticated,
  getAuthToken,
  getBaseURL,
} = apiClient;