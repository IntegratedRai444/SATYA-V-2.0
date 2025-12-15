import { authService } from '@/services/auth';

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

interface ApiRequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  data?: unknown;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  skipAuth?: boolean;
  signal?: AbortSignal;
  isRetry?: boolean;
}

interface PendingRequest {
  promise: Promise<Response>;
  controller: AbortController;
  timestamp: number;
}

// Constants
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const DEFAULT_RETRIES = 2;
const DEFAULT_RETRY_DELAY = 1000; // 1 second
const REQUEST_CACHE = new Map<string, PendingRequest>();
const CACHE_TTL = 1000 * 60 * 5; // 5 minutes

// Clean up old pending requests
const cleanupStaleRequests = () => {
  const now = Date.now();
  for (const [key, request] of REQUEST_CACHE.entries()) {
    if (now - request.timestamp > CACHE_TTL) {
      request.controller.abort('Request timed out');
      REQUEST_CACHE.delete(key);
    }
  }
};

// Set up periodic cleanup
setInterval(cleanupStaleRequests, CACHE_TTL / 2);

export class ApiError extends Error {
  constructor(
    public status: number,
    public code?: string,
    public details?: unknown,
    public isAuthError = false
  ) {
    super(`API Error: ${status}`);
    this.name = 'ApiError';
  }
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;
  private isRefreshing = false;
  private refreshPromise: Promise<{ access_token: string } | null> | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, ''); // Remove trailing slashes
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
  }

  private async refreshToken() {
    if (!this.isRefreshing) {
      this.isRefreshing = true;
      this.refreshPromise = authService.refreshToken()
        .finally(() => {
          this.isRefreshing = false;
          this.refreshPromise = null;
        });
    }
    return this.refreshPromise;
  }

  private async parseError(response: Response): Promise<{
    code?: string;
    message?: string;
    details?: unknown;
  }> {
    const contentType = response.headers.get('content-type');
    
    if (contentType?.includes('application/json')) {
      try {
        const errorData = await response.json();
        return {
          code: errorData.code || `HTTP_${response.status}`,
          message: errorData.message || response.statusText,
          details: errorData.details
        };
      } catch {
        // If we can't parse JSON, return a generic error
      }
    }

    // For non-JSON responses, try to get text
    const text = await response.text();
    if (text) {
      return {
        code: `HTTP_${response.status}`,
        message: text,
        details: text
      };
    }

    // If we can't get text, return a minimal error
    return {
      code: `HTTP_${response.status}`,
      message: response.statusText
    };
  }

  private async request<T = unknown>(
    endpoint: string,
    method: HttpMethod = 'GET',
    options: ApiRequestOptions = {}
  ): Promise<T> {
    const isAuthEndpoint = endpoint.includes('/auth/');
    const {
      params = {},
      data,
      headers = {},
      timeout = DEFAULT_TIMEOUT,
      retries = DEFAULT_RETRIES,
      retryDelay = DEFAULT_RETRY_DELAY,
      skipAuth = false,
      signal: externalSignal,
      isRetry = false,
      ...restOptions
    } = options;

    // Create request key for deduplication
    const requestKey = `${method}:${endpoint}:${JSON.stringify(params)}`;
    
    // Check for duplicate request (only for GET requests and not retries)
    if (method === 'GET' && !isRetry && REQUEST_CACHE.has(requestKey)) {
      const { promise } = REQUEST_CACHE.get(requestKey)!;
      return promise.then(response => response.clone().json()) as Promise<T>;
    }

    const controller = new AbortController();
    const signal = externalSignal || controller.signal;
    
    // Set up timeout
    const timeoutId = setTimeout(() => {
      controller.abort('Request timeout');
    }, timeout);

    // Prepare headers
    const requestHeaders: Record<string, string> = {
      ...this.defaultHeaders,
      ...(headers as Record<string, string>)
    };

    // Add auth token if needed
    if (!skipAuth && !isAuthEndpoint) {
      const token = authService.getAuthToken();
      if (token) {
        requestHeaders['Authorization'] = `Bearer ${token}`;
      }
    }

    // Build URL with query parameters
    const url = new URL(endpoint, this.baseUrl);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.append(key, String(value));
      }
    });

    // Prepare request config
    const config: RequestInit = {
      method,
      headers: new Headers(requestHeaders),
      signal,
      ...restOptions,
    };

    // Add request body for non-GET requests
    if (method !== 'GET' && data !== undefined) {
      config.body = JSON.stringify(data);
    }

    // Create request promise with retry logic
    const makeRequest = async (attempt = 0): Promise<Response> => {
      try {
        const response = await fetch(url.toString(), config);

        // Handle 401 Unauthorized - try to refresh token and retry
        if (response.status === 401 && !isAuthEndpoint && !skipAuth && attempt < retries) {
          try {
            // Try to refresh token
            const refreshResponse = await this.refreshToken();
            
            if (refreshResponse?.access_token) {
              // Update auth header with new token
              requestHeaders['Authorization'] = `Bearer ${refreshResponse.access_token}`;
              config.headers = new Headers(requestHeaders);
              
              // Retry the original request with new token
              return makeRequest(attempt + 1);
            }
          } catch (refreshError) {
            // If refresh fails, clear auth and rethrow
            await authService.logout();
            throw new ApiError(401, 'AUTH_REFRESH_FAILED', 'Failed to refresh token', true);
          }
        }

        // Handle other error statuses
        if (!response.ok) {
          const errorData = await this.parseError(response);
          throw new ApiError(
            response.status,
            errorData?.code || 'API_ERROR',
            errorData?.details || 'An unknown error occurred',
            response.status === 401
          );
        }

        return response;
      } catch (error) {
        // Handle network errors and timeouts
        if (error instanceof Error && error.name === 'AbortError' && attempt < retries) {
          // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, retryDelay * (2 ** attempt)));
          return makeRequest(attempt + 1);
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
      }
    };

    // Execute the request
    try {
      const response = await makeRequest();
      
      // Handle empty responses
      const responseText = await response.text();
      if (!responseText) {
        return null as T;
      }
      
      // Parse JSON response
      return JSON.parse(responseText) as T;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Handle network errors
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError(408, 'REQUEST_TIMEOUT', 'Request timed out');
      }
      
      throw new ApiError(0, 'NETWORK_ERROR', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      // Clean up
      REQUEST_CACHE.delete(requestKey);
    }
  }

  // HTTP method implementations
  async get<T = unknown>(
    endpoint: string,
    options: Omit<ApiRequestOptions, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'GET', options);
  }

  async post<T = unknown>(
    endpoint: string,
    data?: unknown,
    options: Omit<ApiRequestOptions, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'POST', { ...options, data });
  }

  async put<T = unknown>(
    endpoint: string,
    data?: unknown,
    options: Omit<ApiRequestOptions, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'PUT', { ...options, data });
  }

  async patch<T = unknown>(
    endpoint: string,
    data?: unknown,
    options: Omit<ApiRequestOptions, 'method' | 'data'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'PATCH', { ...options, data });
  }

  async delete<T = void>(
    endpoint: string,
    options: Omit<ApiRequestOptions, 'method'> = {}
  ): Promise<T> {
    return this.request<T>(endpoint, 'DELETE', options);
  }

  // Cancel all pending requests
  static cancelAllRequests(): void {
    for (const { controller } of REQUEST_CACHE.values()) {
      controller.abort('Request cancelled by user');
    }
    REQUEST_CACHE.clear();
  }
}

// Create a singleton instance
export const apiClient = new ApiClient(import.meta.env.VITE_API_URL || 'http://localhost:8000');

// Export a default instance for easier imports
export default apiClient;

// Export types
export type { ApiRequestOptions };
