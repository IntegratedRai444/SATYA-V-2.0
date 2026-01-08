// Types
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD';

export interface RetryPolicy {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  factor: number;
  retryOn: number[];
  retryIf: (error: unknown) => boolean;
  jitter?: boolean;
}

export interface ApiRequestOptions<T = unknown> {
  headers?: Record<string, string>;
  params?: Record<string, unknown>;
  data?: T;
  signal?: AbortSignal;
  timeout?: number;
  retryPolicy?: Partial<RetryPolicy>;
  cacheTTL?: number;
  deduplicate?: boolean;
  [key: string]: unknown;
}

interface PendingRequest<T = unknown> {
  promise: Promise<T>;
  controller: AbortController;
  timestamp: number;
  retryCount: number;
  url: string;
  method: string;
  cacheTTL?: number;
  retryPolicy: RetryPolicy;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: unknown,
    public retryable = true
  ) {
    super(message);
    this.name = 'ApiError';
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

// Auth service placeholder
const authService = {
  getToken: (): string | null => {
    return typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  },
  refreshToken: async (): Promise<{ access_token: string }> => {
    // This will be replaced with actual refresh token logic
    return { access_token: '' };
  }
};

const CACHE_CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes

class ApiClient {
  private readonly baseUrl: string;
  private requestCache: Map<string, PendingRequest<unknown>>;
  // Cleanup interval for cache management
  private cleanupInterval: ReturnType<typeof setInterval>;
  private isRefreshing = false;
  private refreshSubscribers: Array<(token: string) => void> = [];
  private refreshPromise: Promise<string> | null = null;

  private defaultRetryPolicy: RetryPolicy = {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 30000,
    factor: 2,
    retryOn: [408, 429, 500, 502, 503, 504],
    retryIf: (error: unknown) => {
      if (error instanceof ApiError) {
        return error.retryable !== false;
      }
      return true;
    },
    jitter: true
  };

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    this.requestCache = new Map();
    
    // Initialize cleanup interval
    this.cleanupInterval = setInterval(() => this.cleanupCache(), CACHE_CLEANUP_INTERVAL);
  }

  // Clean up resources when the instance is no longer needed
  public destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.cleanupCache();
  }

  private cleanupCache(): void {
    const now = Date.now();
    const entries = Array.from(this.requestCache.entries());

    for (const [key, request] of entries) {
      const age = now - request.timestamp;
      const maxAge = request.cacheTTL ?? 5 * 60 * 1000; // Default 5 minutes
      if (age > maxAge) {
        request.controller.abort();
        this.requestCache.delete(key);
      }
    }
  }

  private generateCacheKey(
    method: string,
    url: string,
    params?: Record<string, unknown>,
    data?: unknown
  ): string {
    const keyParts = [method, url];

    if (params) {
      try {
        const paramsKey = Object.entries(params)
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
          .join('&');
        keyParts.push(paramsKey);
      } catch (error) {
        console.warn('Failed to stringify params for cache key', error);
      }
    }

    if (data) {
      try {
        const dataKey = JSON.stringify(data);
        keyParts.push(dataKey);
      } catch (error) {
        console.warn('Failed to stringify data for cache key', error);
      }
    }

    return keyParts.join('|');
  }

  private async waitForTokenRefresh(): Promise<string> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.isRefreshing = true;
    this.refreshPromise = (async (): Promise<string> => {
      try {
        const { access_token } = await authService.refreshToken();
        this.refreshSubscribers.forEach(callback => callback(access_token));
        return access_token;
      } finally {
        this.isRefreshing = false;
        this.refreshPromise = null;
        this.refreshSubscribers = [];
      }
    })();

    return this.refreshPromise;
  }

  private async handleTokenRefresh(): Promise<string> {
    if (this.isRefreshing) {
      return new Promise<string>((resolve) => {
        this.refreshSubscribers.push(resolve);
      });
    }
    return this.waitForTokenRefresh();
  }

  // Error parsing utility method used internally
  private parseError(error: unknown): Error {
    if (error instanceof Error) {
      return error;
    }
    return new Error(String(error));
  }

  private calculateRetryDelay(
    retryCount: number,
    initialDelay: number,
    maxDelay: number,
    factor: number,
    jitter: boolean
  ): number {
    const delay = Math.min(initialDelay * Math.pow(factor, retryCount - 1), maxDelay);
    return jitter ? delay * (0.8 + Math.random() * 0.4) : delay;
  }

  private async handleErrorResponse(response: Response): Promise<ApiError> {
    try {
      const errorData = await response.json().catch(() => ({}));
      return new ApiError(
        errorData.message || response.statusText,
        response.status,
        errorData.code,
        errorData.details,
        errorData.retryable
      );
    } catch (e) {
      // Use parseError for consistent error handling
      const error = this.parseError(e);
      return new ApiError(
        response.statusText,
        response.status,
        undefined,
        error,
        [408, 429, 500, 502, 503, 504].includes(response.status)
      );
    }
  }

  private async makeRequest<T>(
    method: HttpMethod,
    endpoint: string,
    options: ApiRequestOptions<T> = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const cacheKey = this.generateCacheKey(method, url, options.params, options.data);
    
    // Check if we have a pending request for this cache key
    if (options.deduplicate !== false && this.requestCache.has(cacheKey)) {
      const pendingRequest = this.requestCache.get(cacheKey)!;
      return pendingRequest.promise as Promise<T>;
    }

    const controller = new AbortController();
    const signal = options.signal || controller.signal;
    
    const executeRequest = async (): Promise<T> => {
      try {
        // Add auth token if available
        let token = authService.getToken();
        
        // If token is expired, try to refresh it
        if (!token && this.isRefreshing) {
          token = await this.handleTokenRefresh();
        }

        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
          ...(token && { Authorization: `Bearer ${token}` }),
          ...options.headers,
        };

        const response = await fetch(url, {
          method,
          headers,
          body: options.data ? JSON.stringify(options.data) : undefined,
          signal,
        });

        // Handle 401 Unauthorized with token refresh
        if (response.status === 401 && !this.isRefreshing) {
          try {
            const newToken = await this.handleTokenRefresh();
            headers.Authorization = `Bearer ${newToken}`;
            
            // Retry the request with the new token
            const retryResponse = await fetch(url, {
              method,
              headers,
              body: options.data ? JSON.stringify(options.data) : undefined,
              signal,
            });

            if (!retryResponse.ok) {
              throw await this.handleErrorResponse(retryResponse);
            }
            return await retryResponse.json() as T;
          } catch (refreshError) {
            throw new ApiError(
              'Failed to refresh token',
              401,
              'TOKEN_REFRESH_FAILED',
              refreshError
            );
          }
        }

        if (!response.ok) {
          throw await this.handleErrorResponse(response);
        }

        return await response.json() as T;
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          throw new ApiError('Request aborted', 0, 'ABORTED', error, false);
        }
        throw error;
      }
    };

    const requestPromise = (async (): Promise<T> => {
      const retryPolicy = {
        ...this.defaultRetryPolicy,
        ...(options.retryPolicy || {}),
      };

      for (let attempt = 0; attempt <= retryPolicy.maxRetries; attempt++) {
        try {
          return await executeRequest();
        } catch (error) {
          const shouldRetry = error instanceof ApiError 
            ? retryPolicy.retryIf(error) && error.retryable !== false
            : retryPolicy.retryIf(error);

          if (attempt === retryPolicy.maxRetries || !shouldRetry) {
            throw error;
          }

          const delay = this.calculateRetryDelay(
            attempt + 1,
            retryPolicy.initialDelay,
            retryPolicy.maxDelay,
            retryPolicy.factor,
            retryPolicy.jitter ?? true
          );

          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }

      throw new Error('Max retries exceeded');
    })();

    // Store the request in cache if deduplication is enabled
    if (options.deduplicate !== false) {
      this.requestCache.set(cacheKey, {
        promise: requestPromise,
        controller,
        timestamp: Date.now(),
        retryCount: 0,
        url,
        method,
        cacheTTL: options.cacheTTL,
        retryPolicy: {
          ...this.defaultRetryPolicy,
          ...(options.retryPolicy || {}),
        },
      });

      // Remove from cache when the request completes or fails
      requestPromise
        .catch(() => {})
        .finally(() => this.requestCache.delete(cacheKey));
    }

    return requestPromise;
  }

  public async request<T = unknown, D = unknown>(
    endpoint: string,
    method: HttpMethod = 'GET',
    options: Omit<ApiRequestOptions<D>, 'method'> = {}
  ): Promise<T> {
    return this.makeRequest<T>(method, endpoint, options as ApiRequestOptions<T>);
  }

  /**
   * Make a GET request
   */
  public async get<T = unknown>(
    endpoint: string,
    params?: Record<string, unknown>,
    options: Omit<ApiRequestOptions<never>, 'method' | 'body' | 'data' | 'params'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'GET', {
      ...options,
      params,
    });
  }

  public async post<T = unknown, D = unknown>(
    endpoint: string,
    data?: D,
    options: Omit<ApiRequestOptions<D>, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'POST', { ...options, data } as ApiRequestOptions<T>);
  }

  public async put<T = unknown, D = unknown>(
    endpoint: string,
    data?: D,
    options: Omit<ApiRequestOptions<D>, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'PUT', { ...options, data } as ApiRequestOptions<T>);
  }

  public async patch<T = unknown, D = unknown>(
    endpoint: string,
    data?: D,
    options: Omit<ApiRequestOptions<D>, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'PATCH', { ...options, data } as ApiRequestOptions<T>);
  }

  public async delete<T = void>(
    endpoint: string,
    options: Omit<ApiRequestOptions<never>, 'method'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'DELETE', options);
  }
}

// Create a singleton instance with environment variable
const getApiBaseUrl = (): string => {
  if (typeof process !== 'undefined' && process.env.VITE_API_URL) {
    return process.env.VITE_API_URL;
  }
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL as string;
  }
  throw new Error('VITE_API_URL environment variable is not set');
};

// Create and export the API client instance
const createApiClient = () => {
  const client = new ApiClient(getApiBaseUrl());
  
  // Clean up on page unload
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
      client.destroy();
    });
  }
  
  return client;
};

export const apiClient = createApiClient();

export default apiClient;

// Global type declarations
declare global {

  interface Window {
    apiClient: typeof apiClient;
  }
}

// Expose to window for debugging
if (typeof window !== 'undefined') {
  window.apiClient = apiClient;
}
