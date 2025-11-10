import axios, { AxiosInstance, AxiosResponse } from 'axios';
import FormData from 'form-data';
import { logger } from '../../config';

interface AuthCredentials {
  username: string;
  password: string;
  email?: string;
  fullName?: string;
}

interface AnalysisOptions {
  sensitivity?: 'low' | 'medium' | 'high';
  includeDetails?: boolean;
  async?: boolean;
}

class TestApiClient {
  private client: AxiosInstance;
  private authToken: string | null = null;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:3000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: `${baseURL}/api`,
      timeout: 30000,
      validateStatus: () => true // Don't throw on HTTP errors
    });

    // Add request interceptor to include auth token
    this.client.interceptors.request.use((config) => {
      if (this.authToken) {
        config.headers.Authorization = `Bearer ${this.authToken}`;
      }
      return config;
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        logger.debug('API Response', {
          method: response.config.method?.toUpperCase(),
          url: response.config.url,
          status: response.status,
          duration: Date.now() - (response.config as any).startTime
        });
        return response;
      },
      (error) => {
        logger.error('API Error', {
          method: error.config?.method?.toUpperCase(),
          url: error.config?.url,
          error: error.message,
          status: error.response?.status
        });
        return Promise.reject(error);
      }
    );
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.authToken = token;
  }

  /**
   * Get current auth token
   */
  getAuthToken(): string | null {
    return this.authToken;
  }

  /**
   * Check if authenticated
   */
  isAuthenticated(): boolean {
    return !!this.authToken;
  }

  /**
   * Register a new user
   */
  async register(credentials: AuthCredentials): Promise<any> {
    const response = await this.client.post('/auth/register', credentials);
    
    if (response.data.success && response.data.token) {
      this.setAuthToken(response.data.token);
    }
    
    return this.handleResponse(response);
  }

  /**
   * Login user
   */
  async login(username: string, password: string): Promise<any> {
    const response = await this.client.post('/auth/login', {
      username,
      password
    });
    
    if (response.data.success && response.data.token) {
      this.setAuthToken(response.data.token);
    }
    
    return this.handleResponse(response);
  }

  /**
   * Logout user
   */
  async logout(): Promise<any> {
    const response = await this.client.post('/auth/logout');
    this.authToken = null;
    return this.handleResponse(response);
  }

  /**
   * Get auth status
   */
  async getAuthStatus(): Promise<any> {
    const response = await this.client.get('/auth/status');
    return this.handleResponse(response);
  }

  /**
   * Refresh auth token
   */
  async refreshToken(): Promise<any> {
    const response = await this.client.post('/auth/refresh');
    
    if (response.data.success && response.data.token) {
      this.setAuthToken(response.data.token);
    }
    
    return this.handleResponse(response);
  }

  /**
   * Analyze image
   */
  async analyzeImage(file: File, options: AnalysisOptions = {}): Promise<any> {
    const formData = new FormData();
    formData.append('image', file.stream(), {
      filename: file.name,
      contentType: file.type
    });
    formData.append('options', JSON.stringify(options));

    const response = await this.client.post('/analysis/image', formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 60000 // 1 minute for analysis
    });

    return this.handleResponse(response);
  }

  /**
   * Analyze video
   */
  async analyzeVideo(file: File, options: AnalysisOptions = {}): Promise<any> {
    const formData = new FormData();
    formData.append('video', file.stream(), {
      filename: file.name,
      contentType: file.type
    });
    formData.append('options', JSON.stringify(options));

    const response = await this.client.post('/analysis/video', formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 120000 // 2 minutes for video analysis
    });

    return this.handleResponse(response);
  }

  /**
   * Analyze audio
   */
  async analyzeAudio(file: File, options: AnalysisOptions = {}): Promise<any> {
    const formData = new FormData();
    formData.append('audio', file.stream(), {
      filename: file.name,
      contentType: file.type
    });
    formData.append('options', JSON.stringify(options));

    const response = await this.client.post('/analysis/audio', formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 90000 // 1.5 minutes for audio analysis
    });

    return this.handleResponse(response);
  }

  /**
   * Analyze multimodal
   */
  async analyzeMultimodal(
    files: { image?: File; video?: File; audio?: File }, 
    options: AnalysisOptions = {}
  ): Promise<any> {
    const formData = new FormData();
    
    if (files.image) {
      formData.append('image', files.image.stream(), {
        filename: files.image.name,
        contentType: files.image.type
      });
    }
    
    if (files.video) {
      formData.append('video', files.video.stream(), {
        filename: files.video.name,
        contentType: files.video.type
      });
    }
    
    if (files.audio) {
      formData.append('audio', files.audio.stream(), {
        filename: files.audio.name,
        contentType: files.audio.type
      });
    }
    
    formData.append('options', JSON.stringify(options));

    const response = await this.client.post('/analysis/multimodal', formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 180000 // 3 minutes for multimodal analysis
    });

    return this.handleResponse(response);
  }

  /**
   * Get analysis status
   */
  async getAnalysisStatus(analysisId: string): Promise<any> {
    const response = await this.client.get(`/analysis/${analysisId}/status`);
    return this.handleResponse(response);
  }

  /**
   * Get analysis results
   */
  async getAnalysisResults(analysisId: string): Promise<any> {
    const response = await this.client.get(`/analysis/${analysisId}/results`);
    return this.handleResponse(response);
  }

  /**
   * Get analysis history
   */
  async getAnalysisHistory(params: { limit?: number; offset?: number; type?: string } = {}): Promise<any> {
    const response = await this.client.get('/analysis/history', { params });
    return this.handleResponse(response);
  }

  /**
   * Get dashboard stats
   */
  async getDashboardStats(): Promise<any> {
    const response = await this.client.get('/dashboard/stats');
    return this.handleResponse(response);
  }

  /**
   * Get health status
   */
  async getHealth(): Promise<any> {
    const response = await this.client.get('/health');
    return this.handleResponse(response);
  }

  /**
   * Get detailed health status
   */
  async getDetailedHealth(): Promise<any> {
    const response = await this.client.get('/health/detailed');
    return this.handleResponse(response);
  }

  /**
   * Get system metrics
   */
  async getMetrics(): Promise<any> {
    const response = await this.client.get('/health/metrics');
    return this.handleResponse(response);
  }

  /**
   * Handle API response
   */
  private handleResponse(response: AxiosResponse): any {
    const { status, data } = response;

    // Log response for debugging
    logger.debug('API Response handled', {
      status,
      success: data.success,
      hasData: !!data.data,
      hasError: !!data.error
    });

    // Return the response data with status info
    return {
      ...data,
      _httpStatus: status,
      _success: status >= 200 && status < 300
    };
  }

  /**
   * Wait for analysis completion
   */
  async waitForAnalysisCompletion(
    analysisId: string, 
    timeout: number = 120000,
    pollInterval: number = 2000
  ): Promise<any> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const status = await this.getAnalysisStatus(analysisId);
      
      if (status.data?.status === 'completed') {
        return await this.getAnalysisResults(analysisId);
      }
      
      if (status.data?.status === 'failed') {
        throw new Error(`Analysis failed: ${status.data.error || 'Unknown error'}`);
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Analysis timeout after ${timeout}ms`);
  }

  /**
   * Create a test user
   */
  async createTestUser(suffix: string = ''): Promise<{ username: string; password: string; token: string }> {
    const timestamp = Date.now();
    const username = `testuser${suffix}_${timestamp}`;
    const password = 'TestPassword123!';
    const email = `${username}@test.com`;

    const result = await this.register({
      username,
      password,
      email,
      fullName: `Test User ${suffix}`
    });

    if (!result.success) {
      throw new Error(`Failed to create test user: ${result.message}`);
    }

    return {
      username,
      password,
      token: result.token
    };
  }

  /**
   * Cleanup test user
   */
  async cleanupTestUser(username: string): Promise<void> {
    // This would require an admin endpoint to delete users
    // For now, we'll just log the cleanup attempt
    logger.info('Test user cleanup requested', { username });
  }
}

// Export singleton instance
export const apiClient = new TestApiClient();