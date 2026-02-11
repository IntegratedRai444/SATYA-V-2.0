import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
  AxiosPromise
} from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { getAccessToken } from "../auth/getAccessToken"

type RetryConfig = {
  retries?: number;
  retryDelay?: number;
  retryOn?: (error: AxiosError) => boolean;
};

type CacheConfig = {
  enabled?: boolean;
  ttl?: number;
  maxSize?: number;
};

// Extended request config with our custom options
export interface EnhancedRequestConfig extends AxiosRequestConfig {
  id?: string;
  retryCount?: number;
  skipAuth?: boolean;
  _retry?: boolean;
  cacheKey?: string;
  dedupe?: boolean;
  timeout?: number;
  retry?: RetryConfig;
  cache?: CacheConfig | boolean;
  metadata?: {
    startTime?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

interface PendingRequest {
  id: string;
  url: string;
  method: string;
  params: unknown;
  data: unknown;
  controller: AbortController;
  timestamp: number;
  cacheKey: string;
  promise: Promise<AxiosResponse>;
  resolve: (value: AxiosResponse) => void;
  reject: (reason?: unknown) => void;
}

// Removed unused QueuedRequest type

class EnhancedApiClient {
  private handleRequest = async (config: InternalAxiosRequestConfig): Promise<InternalAxiosRequestConfig> => {
    const enhancedConfig = config as EnhancedRequestConfig;
    
    // Add auth token if available and not skipped
    if (!enhancedConfig.skipAuth) {
      const token = await this.getAuthToken();
      console.log("Enhanced client auth token:", token ? "Bearer [REDACTED]" : "null");
      
      if (token) {
        config.headers = {
          ...config.headers,
          'Authorization': `Bearer ${token}`
        } as any;
      } else {
        console.warn("No auth token available for protected request");
        config.headers = {
          ...config.headers,
          'Authorization': null
        } as any;
      }
    }

    // Add request ID for tracking
    const requestId = uuidv4();
    config.headers['X-Request-Id'] = requestId;

    return config;
  };

  private handleRequestError(error: AxiosError): Promise<never> {
    console.error('[API] Request error:', error.message);
    return Promise.reject(error);
  }

  private handleResponse = (response: AxiosResponse): AxiosResponse => {
    // Clean up pending request
    const requestId = response.config.headers['X-Request-Id'] as string | undefined;
    if (requestId) {
      this.removePendingRequest(requestId);
    }
    
    // Cache GET responses
    if (response.config.method?.toUpperCase() === 'GET') {
      const enhancedConfig = response.config as EnhancedRequestConfig;
      if (enhancedConfig.cache !== false) {
        const cacheKey = this.generateCacheKey(enhancedConfig);
        const cacheTtl = typeof enhancedConfig.cache === 'object' 
          ? enhancedConfig.cache.ttl 
          : undefined;
        this.setCache(cacheKey, response.data, cacheTtl);
      }
    }
    
    return response;
  };

  private async handleResponseError(error: AxiosError): Promise<never> {
    if (!error.config) {
      return Promise.reject(error);
    }

    const config = error.config as EnhancedRequestConfig;
    
    // Clean up pending request
    const requestId = config.headers?.['X-Request-Id'] as string | undefined;
    if (requestId) {
      this.removePendingRequest(requestId);
    }

    // Handle network errors
    if (!error.response) {
      console.error('[API] Network error:', error.message);
      return Promise.reject(error);
    }

    const { status } = error.response;

    // Handle 401 Unauthorized (token refresh)
    if (status === 401) {
      // Skip if already refreshing or no refresh token
      if (this.isRefreshing) {
        return new Promise<never>((resolve, reject) => {
          this.requestQueue.push({
            config,
            resolve: () => {
              if (config.headers) {
                config.headers.Authorization = `Bearer ${this.getAuthToken()}`;
              }
              this.client.request(config).then(
                (response) => resolve(response as never),
                (error) => reject(error)
              );
            },
            reject: (err: AxiosError) => reject(err)
          });
        });
      }

      this.isRefreshing = true;

      try {
        const newToken = await this.refreshToken();
        if (newToken) {
          // Update the original request with the new token
          if (config.headers) {
            config.headers.Authorization = `Bearer ${newToken}`;
          }

          // Retry all queued requests with the new token
          this.requestQueue.forEach(({ resolve }) => {
            resolve();
          });

          // Clear the queue
          this.requestQueue = [];

          // Retry the original request
          return this.client.request(config);
        }
      } catch (refreshError) {
        // Token refresh failed, clear auth and redirect to login
        this.clearAuthTokens();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      } finally {
        this.isRefreshing = false;
      }
    }

    // Handle other error statuses
    switch (status) {
      case 400:
        console.error('[API] Bad Request:', error.response?.data);
        break;
      case 403:
        console.error('[API] Forbidden:', error.response?.data);
        break;
      case 404:
        console.error('[API] Not Found:', config.url);
        break;
      case 500:
        console.error('[API] Server Error:', error.response?.data);
        break;
      default:
        console.error(`[API] Error ${status}:`, error.response?.data);
    }

    return Promise.reject(error);
  }

  private removePendingRequest(requestId: string): void {
    // Find and remove the pending request
    for (const [key, request] of this.pendingRequests.entries()) {
      if (request.id === requestId) {
        this.pendingRequests.delete(key);
        break;
      }
    }
  }

  private clearAuthTokens(): void {
    if (typeof document !== 'undefined') {
      document.cookie = 'satyaai_auth_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
      document.cookie = 'satyaai_refresh_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
    }
  }

  private async getAuthToken(): Promise<string | null> {
    return await getAccessToken();
  }

  private refreshToken = async (): Promise<string | null> => {
    try {
      const response = await axios.post<{ accessToken: string }>(
        '/auth/refresh',
        {},
        { withCredentials: true }
      );
      return response.data.accessToken;
    } catch (error) {
      console.error('Failed to refresh token:', error);
      this.clearAuthTokens();
      return null;
    }
  };

  private request<T = unknown>(config: EnhancedRequestConfig): AxiosPromise<T> {
    return this.client.request<T>(config);
  }
  private client: AxiosInstance;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private requestQueue: Array<{
    config: EnhancedRequestConfig;
    resolve: () => void;
    reject: (error: AxiosError) => void;
  }> = [];
  private isRefreshing = false;
  private cache: Map<string, { data: unknown; timestamp: number; ttl: number }> = new Map();
  private maxCacheSize: number = 100;
  
  private defaultCacheConfig: Required<CacheConfig> = {
    enabled: true,
    ttl: 5 * 60 * 1000, // 5 minutes
    maxSize: 100,
  };

  private metrics: {
    totalRequests: number;
    failedRequests: number;
    cacheHits: number;
    cacheMisses: number;
    averageResponseTime: number;
  } = {
    totalRequests: 0,
    failedRequests: 0,
    cacheHits: 0,
    cacheMisses: 0,
    averageResponseTime: 0,
  };

  constructor() {
    const API_BASE_URL = import.meta.env.VITE_API_URL;

    if (!API_BASE_URL) {
      throw new Error('VITE_API_URL environment variable is not set. Please check your .env file');
    }

    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'X-Request-Id': uuidv4(),
      },
      withCredentials: true
    });

    // Set up request interceptor
    this.client.interceptors.request.use(
      this.handleRequest.bind(this),
      this.handleRequestError.bind(this)
    );

    // Set up response interceptor
    this.client.interceptors.response.use(
      this.handleResponse.bind(this),
      this.handleResponseError.bind(this)
    );

    // Clean up stale cache entries periodically
    this.setupCacheCleanup();
  }

  /**
   * Set up periodic cache cleanup
   */
  private setupCacheCleanup(): void {
    setInterval(() => {
      const now = Date.now();
      for (const [key, entry] of this.cache.entries()) {
        if (now - entry.timestamp > entry.ttl) {
          this.cache.delete(key);
        }
      }
    }, 60 * 1000); // Check every minute
  }

  /**
   * Generate a cache key for the request
   */
  private generateCacheKey(config: EnhancedRequestConfig): string {
    if (config.cacheKey) return config.cacheKey;
    
    const { url, method, params, data } = config;
    const keyParts = [
      method?.toUpperCase(),
      url,
      params ? JSON.stringify(params) : '',
      data && typeof data === 'object' ? JSON.stringify(data) : data
    ];
    
    return keyParts.join('|');
  }

  

  /**
   * Add response to cache
   */
  private setCache<T = unknown>(key: string | undefined, data: T, ttl: number = this.defaultCacheConfig.ttl): void {
    if (!key) return;
    // Clean up if cache is too large
    if (this.cache.size >= this.maxCacheSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  

  /**
    };
    
    this.pendingRequests.set(cacheKey, pendingRequest);
    
    // Set up cleanup when the promise settles
    promise.finally(() => {
      this.pendingRequests.delete(cacheKey);
    });
    
    return promise;
  }


  // HTTP method implementations
  public get<T = any>(url: string, config?: EnhancedRequestConfig): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'GET', url });
  }

  public post<T = any, D = any>(
    url: string,
    data?: D,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'POST', url, data });
  }

  public put<T = any, D = any>(
    url: string,
    data?: D,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'PUT', url, data });
  }

  public patch<T = any, D = any>(
    url: string,
    data?: D,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'PATCH', url, data });
  }

  public delete<T = any>(
    url: string,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'DELETE', url });
  }

  /**
   * Get request metrics
   */
  getMetrics() {
    return { ...this.metrics };
  }

  /**
   * Update user profile
   */
  public updateProfile<T = unknown, D = unknown>(
    data?: D,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'PUT', url: '/user/profile', data });
  }

  /**
   * Change user password
   */
  public changePassword<T = unknown>(
    currentPassword: string,
    newPassword: string,
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ 
      ...config, 
      method: 'POST', 
      url: '/auth/change-password',
      data: { currentPassword, newPassword }
    });
  }

  /**
   * Delete user account
   */
  public deleteAccount<T = unknown>(
    config?: EnhancedRequestConfig
  ): AxiosPromise<T> {
    return this.request<T>({ ...config, method: 'DELETE', url: '/user/account' });
  }
  
  /**
   * Clear the request cache
   */
  clearCache(): void {
    this.cache.clear();
  }
  
  /**
   * Cancel all pending requests
   */
  cancelAllRequests(reason = 'Request cancelled by user'): void {
    for (const request of this.pendingRequests.values()) {
      request.controller.abort(reason);
      request.reject(new Error(reason));
    }
    this.pendingRequests.clear();
  }
}

// Create and export a singleton instance
export const apiClient = new EnhancedApiClient();

export default apiClient;
