import axios, { AxiosInstance, AxiosResponse, AxiosRequestConfig } from 'axios';
import { API_CONFIG } from '../config/urls';
import { v4 as uuidv4 } from 'uuid';
import { metrics, trackError as logError } from '@/lib/services/metrics';
import { getAccessToken } from "../auth/getAccessToken";
import { getCsrfToken } from './services/csrfService';
import { getSecureRefreshToken, setSecureAccessToken, clearSecureTokens } from '../auth/secureTokenStorage';

// Export types needed by services
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface DashboardStats {
  totalScans: number;
  manipulatedScans: number;
  averageConfidence: number;
  scanRate: number;
}

export interface ExtendedDashboardStats {
  analyzedMedia: { count: number; growth: string };
  detectedDeepfakes: { count: number; growth: string };
  avgDetectionTime: { time: string; improvement: string };
  detectionAccuracy: { percentage: number; improvement: string };
  dailyActivity: Array<{
    date: string;
    analyses: number;
    deepfakes: number;
  }>;
}

// Types
interface CacheEntry {
  timestamp: number;
  response: AxiosResponse;
  expiry: number;
}

interface RetryConfig {
  retries: number;
  retryDelay: number;
  retryOn: number[];
  maxRetryDelay: number;
}

export interface RequestConfig extends AxiosRequestConfig {
  id?: string;
  retryCount?: number;
  skipCache?: boolean;
  cacheTtl?: number;
  retryConfig?: Partial<RetryConfig>;
  _retry?: boolean;
  metadata?: {
    requestId: string;
    startTime: number;
    cacheKey?: string;
    fromCache?: boolean;
    requestKey?: string;
    source: {
      signal: AbortSignal;
      cancel: (message?: string) => void;
    };
  };
}

// Constants
const API_VERSION = 'v2';
const DEFAULT_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  retries: 3,
  retryDelay: 1000,
  maxRetryDelay: 30000,
  retryOn: [408, 429, 502, 503, 504] // Removed 500 - never retry internal server errors
};

// Caches
const requestCache = new Map<string, CacheEntry>();
const pendingRequests = new Map<string, Promise<AxiosResponse>>();

// Helpers
const generateCacheKey = (config: RequestConfig): string => {
  const { method, url, params, data } = config;
  const serializedParams = params ? JSON.stringify(params) : '';
  const serializedData = data ? (typeof data === 'string' ? data : JSON.stringify(data)) : '';
  return `${method}:${url}:${serializedParams}:${serializedData}`;
};

const getCachedResponse = (key: string): AxiosResponse | null => {
  const entry = requestCache.get(key);
  if (!entry) return null;
  
  if (Date.now() > entry.expiry) {
    requestCache.delete(key);
    return null;
  }
  
  return entry.response;
};

const setCachedResponse = (key: string, response: AxiosResponse, ttl: number = DEFAULT_CACHE_TTL): void => {
  requestCache.set(key, {
    response,
    timestamp: Date.now(),
    expiry: Date.now() + ttl
  });
};

const createRetryDelay = (retryCount: number, config: RetryConfig): number => {
  // Exponential backoff with jitter: delay = base * 2^(retry-1) + random jitter
  const exponentialDelay = config.retryDelay * Math.pow(2, retryCount - 1);
  const jitter = Math.random() * 0.3 * exponentialDelay; // 30% jitter
  const delay = Math.min(exponentialDelay + jitter, config.maxRetryDelay);
  return Math.floor(delay);
};

// Create a cancel token source for request cancellation
const createCancellableSource = () => {
  const controller = new AbortController();
  return {
    signal: controller.signal,
    cancel: (message?: string) => controller.abort(message || 'Request cancelled by user')
  };
};

// Base configuration for axios instances
const createAxiosInstance = (baseURL: string): AxiosInstance => {
  const instance = axios.create({
    baseURL,
    timeout: 30000, // 30 seconds
    headers: {
      'Content-Type': 'application/json',
      'Accept': `application/vnd.satyaai.${API_VERSION}+json`,
      'X-API-Version': API_VERSION,
      'X-Request-Id': uuidv4()
    },
    withCredentials: true,
    signal: createCancellableSource().signal,
    validateStatus: (status) => status >= 200 && status < 300
  });

  return instance;
};

// Create separate instances for auth and analysis
const createEnhancedAxiosInstance = (baseURL: string): AxiosInstance => {
  const instance = createAxiosInstance(baseURL);
  
  // Request interceptor for caching and metrics
  instance.interceptors.request.use(
    async (config: any) => {
      const requestId = uuidv4();
      (config as any).id = requestId;
      (config as any).metadata = {
        requestId,
        startTime: Date.now(),
        source: createCancellableSource()
      };

      // Set appropriate timeout based on request type and content
      if (config.timeout === undefined) {
        const method = config.method?.toUpperCase() || 'GET';
        const isUpload = config.data instanceof FormData;
        const isVideoAnalysis = config.url?.includes('/video') || config.url?.includes('/batch');
        
        if (isUpload) {
          config.timeout = 120000; // 2 minutes for uploads
        } else if (isVideoAnalysis) {
          config.timeout = 300000; // 5 minutes for video analysis
        } else if (method === 'GET') {
          config.timeout = 30000; // 30 seconds for normal GET requests
        } else {
          config.timeout = 60000; // 1 minute for other requests
        }
      }

      // Add CSRF token to all non-GET requests
      if (config.method?.toUpperCase() !== 'GET') {
        try {
          const token = await getCsrfToken();
          if (token) {
            config.headers = config.headers || {};
            config.headers['X-CSRF-Token'] = token;
          }
        } catch (error) {
          console.warn('Failed to fetch CSRF token:', error);
          // Continue without CSRF token in case of failure
        }
      }

      const method = config.method?.toUpperCase() || 'GET';
      const cacheKey = generateCacheKey(config);
      
      // Set request start time for performance tracking
      config.metadata = {
        ...config.metadata,
        startTime: Date.now(),
        requestId,
        cacheKey,
        source: createCancellableSource()
      };

      // Check cache for GET requests
      if (method === 'GET' && !config.skipCache) {
        const cachedResponse = getCachedResponse(cacheKey);
        if (cachedResponse) {
          return {
            ...config,
            adapter: () => Promise.resolve(cachedResponse),
            metadata: {
              ...config.metadata,
              fromCache: true
            }
          };
        }
      }

      // Add request to pending requests map for deduplication
      if (config.id) {
        const pendingRequest = pendingRequests.get(config.id);
        if (pendingRequest) {
          return {
            ...config,
            adapter: () => pendingRequest
          };
        }
      }

      return config;
    },
    (error) => {
      logError(error, 'request_error');
      return Promise.reject(error);
    }
  );

  // Response interceptor for caching and error handling
  instance.interceptors.response.use(
    (response) => {
      const { config } = response;
      const requestConfig = config as RequestConfig;
      const { requestId, cacheKey, startTime } = requestConfig.metadata || {};
      
      // Log successful request
      const duration = startTime ? Date.now() - startTime : 0;
      // Removed noisy request logging for production
      
      // Cache successful GET responses
      if (requestConfig.method?.toUpperCase() === 'GET' && response.status === 200 && cacheKey) {
        setCachedResponse(cacheKey, response, requestConfig.cacheTtl);
      }
      
      // Remove from pending requests
      if (requestConfig.id) {
        pendingRequests.delete(requestConfig.id);
      }
      
      return response;
    },
    async (error) => {
      const { config } = error;
      const requestConfig = config as RequestConfig;
      const { retryCount = 0, retryConfig = {} } = requestConfig;
      const mergedRetryConfig = { ...DEFAULT_RETRY_CONFIG, ...retryConfig };
      
      // Don't retry if there's no config or it's not a retryable error
      if (!config || !mergedRetryConfig.retryOn.includes(error.response?.status)) {
        return Promise.reject(error);
      }
      
      // Never retry multipart/form-data uploads (non-idempotent)
      const isFormData = config.data instanceof FormData;
      const isMultipart = config.headers?.['Content-Type']?.includes('multipart/form-data');
      if (isFormData || isMultipart) {
        return Promise.reject(error);
      }
      
      // Check if we've exceeded max retries
      if (retryCount >= mergedRetryConfig.retries) {
        return Promise.reject(error);
      }
      
      // Calculate retry delay with exponential backoff and jitter
      const delay = createRetryDelay(retryCount + 1, mergedRetryConfig);
      
      // Create a new promise that will resolve after the delay
      await new Promise(resolve => setTimeout(resolve, delay));
      
      // Retry the request
      return instance({
        ...config,
        retryCount: retryCount + 1
      });
    }
  );
  
  return instance;
};

// Create unified API client instance
const apiClient = createEnhancedAxiosInstance(import.meta.env.VITE_API_URL);

// Function to set the auth service reference (kept for backward compatibility)
export const setAuthService = () => {
  // No-op: kept for backward compatibility
};

// Token refresh state management
let isRefreshing = false;
let refreshPromise: Promise<string> | null = null;
const refreshQueue: Array<{ resolve: (token: string) => void; reject: (error: Error) => void }> = [];

// Enhanced token refresh with race condition prevention
const refreshTokenWithQueue = async (): Promise<string> => {
  if (isRefreshing) {
    // If already refreshing, wait for the current refresh to complete
    if (refreshPromise) {
      return refreshPromise;
    }
    
    // Queue this request
    return new Promise((resolve, reject) => {
      refreshQueue.push({ resolve, reject });
    });
  }

  isRefreshing = true;
  
  try {
    refreshPromise = (async () => {
      const refreshToken = getSecureRefreshToken();
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const response = await axios.post(
        `${API_CONFIG.BASE_URL}/auth/refresh-token`,
        { refreshToken },
        { withCredentials: true }
      );
      
      const { accessToken } = response.data;
      setSecureAccessToken(accessToken);
      
      // Resolve all queued requests
      refreshQueue.forEach(({ resolve }) => resolve(accessToken));
      refreshQueue.length = 0;
      
      return accessToken;
    })();
    
    return await refreshPromise;
  } catch (error) {
    // Reject all queued requests
    refreshQueue.forEach(({ reject }) => reject(error as Error));
    refreshQueue.length = 0;
    throw error;
  } finally {
    isRefreshing = false;
    refreshPromise = null;
  }
};
const batchInterval = 100; // 100ms
let batchQueue: Array<{
  key: string;
  config: RequestConfig;
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
}> = [];

const processBatch = async () => {
  if (batchQueue.length === 0) return;
  
  const currentBatch = [...batchQueue];
  batchQueue = [];
  
  try {
    const batchResponses = await Promise.allSettled(
      currentBatch.map(({ config }) => apiClient.request(config))
    );
    
    // Resolve or reject each promise in the batch
    currentBatch.forEach(({ resolve, reject }, index) => {
      const result = batchResponses[index];
      if (result.status === 'fulfilled') {
        const response = {
          ...result.value,
          config: result.value.config as RequestConfig
        };
        resolve(response.data);
      } else {
        reject(result.reason);
      }
    });
  } catch (error) {
    // If the entire batch fails, reject all promises
    currentBatch.forEach(({ reject }) => reject(error));
  }
};

setInterval(processBatch, batchInterval);

// Request interceptor for batching
apiClient.interceptors.request.use(
  async (config: any) => {
    // Skip batching for now to simplify type issues
    // Skip deduplication for non-idempotent methods
    const isIdempotent = ['GET', 'HEAD', 'OPTIONS'].includes(config.method?.toUpperCase() || '');
    
    // Create a unique key for the request
    const requestKey = [
      config.method,
      config.url,
      config.params ? JSON.stringify(config.params) : '',
      config.data ? JSON.stringify(config.data) : ''
    ].join('_');
    
    // Add auth token if available
    const token = await getAccessToken();
    // Removed noisy auth token logging for production
    
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    } else {
      console.error("User session missing - enhanced client requests will be unauthenticated");
    }
    
    // Generate request ID and track start time
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const source = createCancellableSource();
    
    // Store request metadata
    config.headers = config.headers || {};
    config.headers['X-Request-ID'] = requestId;
    config.signal = source.signal;
    
    const requestConfig = {
      ...config,
      metadata: { 
        requestId, 
        startTime,
        source,
        requestKey
      }
    };
    
    // Track request
    metrics.track({
      type: 'api',
      name: 'api.request',
      value: 1,
      metadata: {
        url: config.url,
        signal: (config as any).metadata?.source?.signal,
        method: config.method?.toUpperCase() || 'GET',
        requestId,
        version: API_VERSION
      },
    });
    
    // Store as pending request for deduplication
    if (isIdempotent) {
      const requestPromise = axios({
        ...requestConfig,
        headers: {
          ...requestConfig.headers,
          'X-Request-ID': requestId
        }
      }).finally(() => {
        pendingRequests.delete(requestKey);
      });
      pendingRequests.set(requestKey, requestPromise);
      return requestConfig as AxiosRequestConfig;
    }
    
    return requestConfig;
  },
  (error) => {
    logError(new Error(`Request interceptor error: ${error.message}`), 'request_interceptor');
    return Promise.reject({
      code: 'REQUEST_ERROR',
      message: 'Failed to process request',
      originalError: error
    });
  }
);
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    const config = response.config as RequestConfig;
    const { requestId, startTime } = config.metadata || {};
    const { status, statusText, data } = response;

    if (requestId && startTime) {
      const duration = Date.now() - startTime;

      // Track successful response with all relevant data
      metrics.track({
        type: 'api',
        name: 'api.response',
        value: duration,
        metadata: {
          url: response.config.url,
          method: response.config.method?.toUpperCase(),
          status,
          statusText,
          requestId,
          duration,
          version: API_VERSION,
          fromCache: (response.headers as Record<string, unknown>)['x-cache'] === 'HIT',
          data: data ? JSON.stringify(data).substring(0, 200) : undefined
        },
      });

      // Clean up request cache
      if (config.metadata?.requestKey) {
        pendingRequests.delete(config.metadata.requestKey);
      }
    }

    return response;
  },
  async (error: unknown) => {
    const axiosError = error as { 
      config?: RequestConfig; 
      response?: { status?: number; data?: unknown; headers?: unknown }; 
      request?: unknown;
      message?: string;
      _retry?: boolean 
    };
    const originalRequest = axiosError.config as RequestConfig;
    const { requestId } = originalRequest?.metadata || {};

    // If error is not 401 or we've already retried, handle and reject
    if (axiosError.response?.status !== 401 || originalRequest._retry) {
      // Log error details
      if (axiosError.response) {
        console.error('API Error Response:', {
          status: axiosError.response.status,
          data: axiosError.response.data,
          headers: axiosError.response.headers,
          requestId,
        });
      } else if (axiosError.request) {
        console.error('API Request Error:', {
          request: axiosError.request,
          requestId,
        });
      } else {
        console.error('API Error:', {
          message: axiosError.message,
          requestId,
        });
      }
      return Promise.reject(axiosError);
    }

    // Skip refresh for login/register endpoints
    if (originalRequest.url?.includes('/auth/login') || 
        originalRequest.url?.includes('/auth/register')) {
      return Promise.reject(axiosError);
    }

    originalRequest._retry = true;
    
    try {
      // Try to refresh the token with race condition prevention
      const accessToken = await refreshTokenWithQueue();
      
      // Update the authorization header
      originalRequest.headers = originalRequest.headers || {};
      originalRequest.headers.Authorization = `Bearer ${accessToken}`;
      
      // Retry the original request
      return apiClient(originalRequest);
    } catch (refreshError) {
      // If refresh fails, log out the user
      if (refreshError instanceof Error) {
        logError(refreshError, 'auth.refresh_token', {
          requestId,
          url: originalRequest?.url,
        });
      } else {
        logError(new Error(String(refreshError)), 'auth.refresh_token', {
          requestId,
          url: originalRequest?.url,
        });
      }
      console.error('Session expired. Please log in again.');
      // Clear tokens securely
      clearSecureTokens();
      window.location.href = '/login';
    }
  }
);

// Cancel all pending requests is now available as api.cancelAllRequests()

export const api = {
  get: async <T = unknown>(
    url: string, 
    params?: Record<string, unknown>, 
    config: Partial<AxiosRequestConfig> = {}
  ): Promise<T> => {
    const response = await apiClient.get(url, { ...config, params });
    return response.data;
  },

  post: async <T = unknown, D = unknown>(
    url: string, 
    data?: D, 
    config: Partial<AxiosRequestConfig> = {}
  ): Promise<T> => {
    const response = await apiClient.post(url, data, config);
    return response.data;
  },

  put: async <T = unknown, D = unknown>(
    url: string, 
    data?: D, 
    config: Partial<AxiosRequestConfig> = {}
  ): Promise<T> => {
    const response = await apiClient.put(url, data, config);
    return response.data;
  },

  delete: async <T = unknown>(
    url: string, 
    config: Partial<AxiosRequestConfig> = {}
  ): Promise<T> => {
    const response = await apiClient.delete(url, config);
    return response.data;
  },

// Cancel a specific request by ID
cancelRequest: (requestId: string, message: string = 'Request cancelled'): boolean => {
  const pendingRequest = pendingRequests.get(requestId);
  if (pendingRequest) {
    const source = createCancellableSource();
    source.cancel(message);
    pendingRequests.delete(requestId);
    return true;
  }
  return false;
  },

  // Cancel all pending requests
  cancelAllRequests: (message: string = 'All requests cancelled'): void => {
    pendingRequests.forEach((_, requestId) => {
      api.cancelRequest(requestId, message);
    });
  },

  // Clear the request cache
  clearCache: (): void => {
    requestCache.clear();
  },
  
  // Get cache statistics
  getCacheStats: () => ({
    size: requestCache.size,
    keys: Array.from(requestCache.keys())
  }),
  
  // Error handling utility
  handleError: (error: unknown): never => {
    if (axios.isCancel(error)) {
      throw new Error('Request was cancelled');
    }
    
    const axiosError = error as { response?: { status?: number; data?: unknown }; data?: unknown; message?: string; request?: unknown };
    
    if (axiosError.response) {
      const { status, data } = axiosError.response;
      const message = (data as { message?: string })?.message || axiosError.message || 'An error occurred';
      
      const errorWithStatus = new Error(message) as Error & { status?: number; data?: unknown };
      errorWithStatus.status = status;
      errorWithStatus.data = data;
      
      throw errorWithStatus;
    } else if (axiosError.request) {
      throw new Error('Network error - no response received');
    } else {
      throw new Error(axiosError.message || 'Unknown error');
    }
  }
};

// Client instances (unified to single client)
export const clients = {
  main: apiClient,
  auth: apiClient,  // All use the same unified client
  analysis: apiClient
};

// Response creation is handled by Axios internally

// Export the API client
export { apiClient };
export default api;
