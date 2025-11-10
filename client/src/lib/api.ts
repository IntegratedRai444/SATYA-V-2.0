/**
 * SatyaAI API Client
 * Handles all communication with the backend API
 */

import axios, { type AxiosInstance } from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
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
    

    // Load auth token from localStorage
    this.authToken = localStorage.getItem('satyaai_token');
    if (this.authToken) {
      this.setAuthToken(this.authToken);
    }

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        
        // Add CORS headers
        config.headers['Access-Control-Allow-Origin'] = window.location.origin;
        config.headers['Access-Control-Allow-Credentials'] = 'true';
        
        // Ensure withCredentials is set
        config.withCredentials = true;
        
        // Enhanced logging in development
        if (import.meta.env.DEV) {
          console.group(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`);
          console.log('📤 Config:', {
            url: config.url,
            method: config.method,
            headers: config.headers,
            timeout: config.timeout,
            baseURL: config.baseURL,
            withCredentials: config.withCredentials
          });
          if (config.data) {
            console.log('📦 Data:', config.data);
          }
          console.groupEnd();
        }
        
        return config;
      },
      (error) => {
        if (import.meta.env.DEV) {
          console.error('❌ Request interceptor error:', error);
          if (error.response) {
            console.error('Response error:', {
              status: error.response.status,
              statusText: error.response.statusText,
              data: error.response.data,
              headers: error.response.headers
            });
          } else if (error.request) {
            console.error('No response received:', error.request);
          } else {
            console.error('Error setting up request:', error.message);
          }
        }
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        // Enhanced logging in development
        if (import.meta.env.DEV) {
          console.group(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`);
          console.log('📥 Response:', {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers,
            data: response.data,
            config: {
              withCredentials: response.config.withCredentials,
              baseURL: response.config.baseURL,
              url: response.config.url,
              method: response.config.method
            }
          });
          console.groupEnd();
        }
        return response;
      },
      (error) => {
        // Enhanced error logging in development
        if (import.meta.env.DEV) {
          console.group(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url}`);
          
          // Log CORS specific issues
          if (error.message?.includes('Network Error') && !error.response) {
            console.error('CORS or Network Error:', {
              message: 'This is likely a CORS issue. Check if the server is running and CORS is properly configured.',
              details: 'Make sure the server is running and the Access-Control-Allow-Origin header is set correctly.'
            });
          }
          
          console.error('Error details:', {
            message: error.message,
            code: error.code,
            status: error.response?.status,
            statusText: error.response?.statusText,
            data: error.response?.data,
            config: {
              url: error.config?.url,
              baseURL: error.config?.baseURL,
              method: error.config?.method,
              withCredentials: error.config?.withCredentials,
              headers: error.config?.headers
            }
          });
          
          // Specific error diagnostics
          if (error.code === 'ECONNREFUSED') {
            console.warn('🔧 Connection refused - Backend server may not be running');
            console.log('💡 Try running: npm run dev (in another terminal)');
          } else if (error.code === 'NETWORK_ERROR' || error.message?.includes('Network Error')) {
            console.warn('🌐 Network error - Check your internet connection and CORS configuration');
            console.log('💡 Check if the backend server is running and CORS is properly configured');
          } else if (error.response?.status === 404) {
            console.warn('🔍 Endpoint not found - Check if the API route exists');
          } else if (error.response?.status === 401) {
            console.warn('🔐 Unauthorized - Check if you need to log in');
          } else if (error.response?.status === 403) {
            console.warn('🚫 Forbidden - You do not have permission to access this resource');
          } else if (error.response?.status >= 500) {
            console.warn('🚨 Server error - Check backend logs');
          }
          
          // Log CORS specific headers if available
          if (error.response?.headers) {
            console.log('Response Headers:', error.response.headers);
          }
          
          console.groupEnd();
        }
        
        // Handle auth errors
        if (error.response?.status === 401) {
          console.log('🔐 Authentication failed, clearing auth and redirecting...');
          this.clearAuth();
          // Only redirect if not already on login page
          if (window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
        }
        
        // Handle CORS errors
        if (error.message?.includes('Network Error') && !error.response) {
          error.isCorsError = true;
        }
        
        // Handle rate limiting
        if (error.response?.status === 429) {
          console.warn('⏱️ Rate limit exceeded. Please try again later.');
        }
        
        return Promise.reject(error);
      }
    );
  }

  // Auth methods
  setAuthToken(token: string) {
    this.authToken = token;
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    localStorage.setItem('satyaai_token', token);
  }

  clearAuth() {
    this.authToken = null;
    delete this.client.defaults.headers.common['Authorization'];
    localStorage.removeItem('satyaai_token');
  }

  // Authentication endpoints
  async login(username: string, password: string): Promise<AuthResponse> {
    try {
      console.log('Making login request to:', `${API_BASE_URL}/api/auth/login`);
      const response = await this.client.post('/api/auth/login', {
        username,
        password,
      });
      
      console.log('Login response:', response.data);
      
      if (response.data.success && response.data.token) {
        this.setAuthToken(response.data.token);
      }
      
      return response.data;
    } catch (error: any) {
      console.error('Login request failed:', error.response?.data || error.message);
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
      console.error('Registration request failed:', error.response?.data || error.message);
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
      console.warn('Logout request failed:', error);
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

  // Health and status endpoints
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/api/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<any> {
    const response = await this.client.get('/api/health/detailed');
    return response.data;
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
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('API Error Response:', {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data,
          headers: error.response.headers
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
        // The request was made but no response was received
        console.error('No response received:', error.request);
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
    console.error('API Error:', error);
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
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
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
        error: error.message
      };
    }
  }

  async retryConnection(maxRetries: number = 3): Promise<boolean> {
    console.log(`🔄 Retrying connection (attempt ${this.connectionStatus.retryCount + 1}/${maxRetries})`);
    
    for (let i = 0; i < maxRetries; i++) {
      const health = await this.checkBackendHealth();
      if (health.connected) {
        return true;
      }
      
      // Exponential backoff: 1s, 2s, 4s
      const delay = Math.pow(2, i) * 1000;
      console.log(`⏳ Waiting ${delay}ms before next retry...`);
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
    
    // Configuration check
    const config = this.validateConfiguration();
    console.log('⚙️ Configuration:', {
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      valid: config.valid,
      issues: config.issues
    });
    
    // Environment variables
    console.log('🌍 Environment:', {
      NODE_ENV: import.meta.env.MODE,
      DEV: import.meta.env.DEV,
      VITE_API_URL: import.meta.env.VITE_API_URL,
      VITE_BYPASS_AUTH: import.meta.env.VITE_BYPASS_AUTH,
      VITE_DEMO_MODE: import.meta.env.VITE_DEMO_MODE
    });
    
    // Connection test
    console.log('🔍 Testing backend connection...');
    const health = await this.checkBackendHealth();
    console.log('🏥 Backend health:', health);
    
    // Auth status
    console.log('🔐 Authentication:', {
      hasToken: !!this.authToken,
      token: this.authToken ? `${this.authToken.substring(0, 10)}...` : null
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