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
        const { config, response } = error;
        
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
    // Don't retry if no config or max retries exceeded
    if (!error.config || (error.config.retryCount || 0) >= this.retryConfig.maxRetries) {
      return false;
    }

    // Retry on network errors or 5xx responses
    if (!error.response) {
      return true; // Network error
    }

    // Retry on server errors and rate limits
    return (
      error.response.status >= 500 || 
      error.response.status === 429 || // Too Many Requests
      error.response.status === 408    // Request Timeout
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
        // Use server-provided error if available
        return {
          error: response.data?.error || 'Request failed',
          code: response.status,
          details: response.data?.details || {
            status: response.status,
            statusText: response.statusText,
            url: error.config?.url,
          },
        };
      }

      // Network error or timeout
      if (error.code === 'ECONNABORTED') {
        return {
          error: 'Request timeout',
          code: 408,
          details: { message: 'The request timed out' },
        };
      }

      // Other axios errors
      return {
        error: error.message || 'Network error',
        code: 0,
        details: { message: 'A network error occurred' },
      };
    }

    // Non-axios error
    return {
      error: error.message || 'Unknown error',
      code: 500,
      details: { message: 'An unexpected error occurred' },
    };
  }

  /**
   * Make a request to the Python service
   */
  private async request<T = any>(config: AxiosRequestConfig): Promise<AxiosResponse<ApiResponse<T>>> {
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

  public async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'GET', url });
    return this.handleResponse(response);
  }

  public async post<T = any, D = any>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'POST', url, data });
    return this.handleResponse(response);
  }

  public async put<T = any, D = any>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'PUT', url, data });
    return this.handleResponse(response);
  }

  public async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.request<T>({ ...config, method: 'DELETE', url });
    return this.handleResponse(response);
  }

  private handleResponse<T>(response: AxiosResponse<ApiResponse<T>>): T {
    if (response.data && response.data.success) {
      return response.data.data as T;
    }
    
    const error = new Error(response.data?.error?.message || 'Request failed');
    (error as any).code = response.data?.error?.code || 500;
    (error as any).details = response.data?.error?.details || {};
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
