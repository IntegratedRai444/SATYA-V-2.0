import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';

// Extend AxiosRequestConfig to include retryCount
declare module 'axios' {
  export interface AxiosRequestConfig {
    retryCount?: number;
  }
}
import { logger } from '../config/logger';
import { pythonConfig } from '../config/python-config';
import { PythonApiError, RetryConfig, ApiResponse } from '../types/python-api';


// Default configuration with increased timeout for ML operations
const DEFAULT_CONFIG: AxiosRequestConfig = {
  baseURL: pythonConfig.apiUrl,
  timeout: pythonConfig.requestTimeout,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': pythonConfig.apiKey,
  },
  validateStatus: (status) => status < 500, // Don't throw for 4xx errors
};

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: pythonConfig.maxRetries,
  retryDelay: pythonConfig.retryDelay,
  maxRetryDelay: pythonConfig.maxRetryDelay,
  retryBackoffFactor: pythonConfig.retryBackoffFactor,
};

class PythonHttpBridge {
  private client: AxiosInstance;
  private isAvailable: boolean = false;
  private lastCheck: number = 0;
  private readonly CHECK_INTERVAL = 30000; // 30 seconds
  private readonly retryConfig: RetryConfig;

  constructor(config: AxiosRequestConfig = {}, retryConfig: Partial<RetryConfig> = {}) {
    this.retryConfig = { ...DEFAULT_RETRY_CONFIG, ...retryConfig };
    
    this.client = axios.create({
      ...DEFAULT_CONFIG,
      ...config,
    });

    // Add request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = process.env.PYTHON_SERVICE_TOKEN;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(this.normalizeError(error))
    );

    // Add response interceptor for error handling and retries
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError<PythonApiError>) => {
        const { config } = error;
        
        // If no config or max retries exceeded, reject
        if (!config || !this.shouldRetry(error)) {
          return Promise.reject(this.normalizeError(error));
        }

        // Set up retry config
        config.retryCount = (config.retryCount || 0) + 1;
        const delay = this.calculateRetryDelay(config.retryCount);

        // Add jitter to prevent thundering herd
        const jitter = Math.random() * 1000;
        
        // Wait for the delay before retrying
        await new Promise(resolve => setTimeout(resolve, delay + jitter));

        // Retry the request
        return this.client(config);
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    const retryCount = error.config?.retryCount || 0;
    
    // Don't retry if we've exceeded max retries
    if (retryCount >= this.retryConfig.maxRetries) {
      return false;
    }

    // Don't retry on file upload endpoints (prevent duplicate uploads)
    const isUploadEndpoint = error.config?.url?.includes('/upload') || 
                           error.config?.url?.includes('/analysis/');

    // Retry only on network errors and 5xx server errors (not on timeouts for uploads)
    return (
      !error.response || // Network errors
      (error.response.status >= 500 && !isUploadEndpoint) || // Server errors (except uploads)
      (error.response.status === 429 && !isUploadEndpoint) // Rate limit (except uploads)
    );
  }

  private calculateRetryDelay(retryCount: number): number {
    const delay = Math.min(
      this.retryConfig.retryDelay * Math.pow(this.retryConfig.retryBackoffFactor, retryCount - 1),
      this.retryConfig.maxRetryDelay
    );
    return delay;
  }

  private normalizeError(error: AxiosError<PythonApiError> | Error): PythonApiError {
    if (axios.isAxiosError(error)) {
      const { response } = error as AxiosError<PythonApiError>;
      
      if (response) {
        const status = response.status;
        
        // Standardize error codes based on HTTP status
        let code: string;
        let message: string;
        
        switch (status) {
          case 400:
            code = 'INVALID_REQUEST';
            message = 'Invalid request to AI service';
            break;
          case 401:
            code = 'PYTHON_AUTH_ERROR';
            message = 'AI service authentication failed';
            break;
          case 403:
            code = 'PYTHON_FORBIDDEN';
            message = 'AI service access forbidden';
            break;
          case 404:
            code = 'PYTHON_NOT_FOUND';
            message = 'AI service endpoint not found';
            break;
          case 422:
            code = 'VALIDATION_ERROR';
            message = 'Invalid data provided to AI service';
            break;
          case 500:
            code = 'PYTHON_ERROR';
            message = 'AI service internal error';
            break;
          case 502:
            code = 'PYTHON_DOWN';
            message = 'AI engine temporarily unavailable';
            break;
          case 503:
            code = 'SERVICE_UNAVAILABLE';
            message = 'AI service temporarily unavailable';
            break;
          case 504:
            code = 'ANALYSIS_TIMEOUT';
            message = 'Analysis request timed out';
            break;
          default:
            code = 'PYTHON_ERROR';
            message = `AI service error (${status})`;
        }
        
        return {
          error: response.data?.error || message,
          code: status,
          details: {
            code,
            message,
            originalError: response.data?.error,
            status,
            statusText: response.statusText,
            url: error.config?.url,
            timestamp: new Date().toISOString()
          },
        };
      }

      // Network error or timeout
      if (error.code === 'ECONNABORTED') {
        return {
          error: 'Analysis timeout',
          code: 504,
          details: {
            code: 'ANALYSIS_TIMEOUT',
            message: 'The analysis request timed out',
            timeout: pythonConfig.requestTimeout,
            timestamp: new Date().toISOString()
          },
        };
      }

      // Connection refused
      if (error.code === 'ECONNREFUSED') {
        return {
          error: 'AI engine unavailable',
          code: 503,
          details: {
            code: 'PYTHON_DOWN',
            message: 'Cannot connect to AI service',
            url: error.config?.url,
            timestamp: new Date().toISOString()
          },
        };
      }

      // Other network errors
      return {
        error: 'Network error',
        code: 0,
        details: {
          code: 'NETWORK_ERROR',
          message: error.message || 'A network error occurred',
          timestamp: new Date().toISOString()
        },
      };
    }

    // Non-axios error
    return {
      error: 'Unexpected error',
      code: 500,
      details: {
        code: 'UNKNOWN_ERROR',
        message: error.message || 'An unexpected error occurred',
        timestamp: new Date().toISOString()
      },
    };
  }

  /**
   * Make a request to the Python service
   */
  private async request<T = unknown>(config: AxiosRequestConfig): Promise<AxiosResponse<ApiResponse<T>>> {
    try {
      const response = await this.client.request<ApiResponse<T>>({
        ...config,
        retryCount: 0, // Initialize retry count
      });
      return response;
    } catch (error) {
      throw this.normalizeError(error as AxiosError<PythonApiError>);
    }
  }

  public async get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'GET', url });
    return this.handleResponse(response);
  }

  public async post<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'POST', url, data });
    return this.handleResponse(response);
  }

  public async put<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'PUT', url, data });
    return this.handleResponse(response);
  }

  public async delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'DELETE', url });
    return this.handleResponse(response);
  }

  private handleResponse<T>(response: AxiosResponse<ApiResponse<T>>): T {
    if (response.data && response.data.success) {
      return response.data.data as T;
    }
    
    const error = new Error(response.data?.error?.message || 'Request failed') as Error & {
      code?: number | string;
      details?: Record<string, unknown>;
    };
    error.code = response.data?.error?.code || 500;
    error.details = response.data?.error?.details || {};
    throw error;
  }

  async checkServiceHealth(): Promise<boolean> {
    const now = Date.now();
    
    // Only check once per CHECK_INTERVAL
    if (now - this.lastCheck < this.CHECK_INTERVAL) {
      return this.isAvailable;
    }

    try {
      const response = await this.request({
        url: '/health',
        method: 'GET',
        timeout: 5000,
      });
      
      this.isAvailable = response.status === 200 && response.data?.success === true;
    } catch (error) {
      logger.error('Python service health check failed', { 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      this.isAvailable = false;
    }

    this.lastCheck = now;
    return this.isAvailable;
  }

  /**
   * Refresh the authentication token
   */
  private async refreshToken(): Promise<void> {
    try {
      const response = await axios.post(
        `${DEFAULT_CONFIG.baseURL}/auth/refresh`,
        {},
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const { token } = response.data;
      // Update the default headers
      this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      // Update the environment variable if needed
      if (process.env.PYTHON_SERVICE_TOKEN) {
        process.env.PYTHON_SERVICE_TOKEN = token;
      }
    } catch (error) {
      logger.error('Failed to refresh Python service token', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw new Error('Failed to refresh authentication token');
    }
  }


  /**
   * Get the current status of the Python service
   */
  getStatus() {
    return {
      isAvailable: this.isAvailable,
      lastCheck: new Date(this.lastCheck).toISOString(),
      baseURL: this.client.defaults.baseURL,
    };
  }
}

// Create a singleton instance
export const pythonBridge = new PythonHttpBridge();

// Export for health checks
export const checkPythonService = (): Promise<boolean> => {
  return pythonBridge.checkServiceHealth();
};

export default pythonBridge;
