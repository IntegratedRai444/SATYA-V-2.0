import axios, { AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { metrics, trackError as logError } from '@/lib/services/metrics';
import { getCsrfToken } from './services/csrfService';

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

export interface RequestConfig extends InternalAxiosRequestConfig {
  id?: string;
  retryCount?: number;
  skipCache?: boolean;
  cacheTtl?: number;
  retryConfig?: Partial<RetryConfig>;
  metadata?: {
    requestId: string;
    startTime: number;
    cacheKey?: string;
    fromCache?: boolean;
    requestKey?: string;
    source?: {
      token: any;
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
  retryOn: [408, 429, 500, 502, 503, 504]
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
  const delay = Math.min(
    config.retryDelay * Math.pow(2, retryCount - 1) + Math.random() * 1000, // Add jitter
    config.maxRetryDelay
  );
  return delay;
};

// Create a cancel token source for request cancellation
const createCancellableSource = () => {
  const source = axios.CancelToken.source();
  return {
    token: source.token,
    cancel: (message?: string) => source.cancel(message || 'Request cancelled by user')
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
    cancelToken: createCancellableSource().token,
    validateStatus: (status) => status >= 200 && status < 300
  });

  return instance;
};

// Create separate instances for auth and analysis
const createEnhancedAxiosInstance = (baseURL: string): AxiosInstance => {
  const instance = createAxiosInstance(baseURL);
  
  // Request interceptor for caching and metrics
  instance.interceptors.request.use(
    async (config: RequestConfig) => {
      const requestId = uuidv4();
      config.id = requestId;
      config.metadata = {
        requestId,
        startTime: Date.now(),
      };

      // Add CSRF token to all non-GET requests
      if (config.method?.toUpperCase() !== 'GET') {
        try {
          const token = await getCsrfToken();
          if (token) {
            config.headers = config.headers || {};
            config.headers['X-CSRF-Token'] = token;
          }
        } catch (error) {
          console.warn('Failed to get CSRF token:', error);
        }
      }

      const method = config.method?.toUpperCase() || 'GET';
      const cacheKey = generateCacheKey(config);
      
      // Set request start time for performance tracking
      config.metadata = {
        ...config.metadata,
        startTime: Date.now(),
        requestId,
        cacheKey
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
      console.log(`[${requestId}] ${requestConfig.method?.toUpperCase()} ${requestConfig.url} ${response.status} (${duration}ms)`);
      
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

// Create separate instances for auth and analysis
const authApiClient = createEnhancedAxiosInstance(import.meta.env.VITE_AUTH_API_URL);
const analysisApiClient = createEnhancedAxiosInstance(import.meta.env.VITE_ANALYSIS_API_URL);

// Function to set the auth service reference (kept for backward compatibility)
export const setAuthService = () => {
  // No-op: kept for backward compatibility
};

// Default export is the analysis client for backward compatibility
const apiClient = createEnhancedAxiosInstance(import.meta.env.VITE_API_URL);

// Request batching
const batchInterval = 100; // 100ms
let batchQueue: Array<{
  key: string;
  config: RequestConfig;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
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

// Process batch queue every batchInterval ms
setInterval(processBatch, batchInterval);

// Request interceptor for batching
apiClient.interceptors.request.use(
  (config: RequestConfig) => {
    // Skip batching for non-GET requests or if explicitly disabled
    const shouldBatch = config.method?.toUpperCase() === 'GET' && config.params?.batch !== false;
    
    if (shouldBatch) {
      return new Promise((resolve, reject) => {
        batchQueue.push({
          key: generateCacheKey(config),
          config,
          resolve: (data: any) => {
            const response: AxiosResponse = {
              data,
              status: 200,
              statusText: 'OK',
              headers: {},
              config: config as InternalAxiosRequestConfig,
              request: {}
            };
            resolve(response as any);
          },
          reject
        });
      });
    }
    
    // Skip deduplication for non-idempotent methods
    const isIdempotent = ['GET', 'HEAD', 'OPTIONS'].includes(config.method?.toUpperCase() || '');
    
    // Create a unique key for the request
    const requestKey = [
      config.method,
      config.url,
      config.params ? JSON.stringify(config.params) : '',
      config.data ? JSON.stringify(config.data) : ''
    ].join('_');
    
    // Check for duplicate requests
    if (isIdempotent && pendingRequests.has(requestKey)) {
      return pendingRequests.get(requestKey)!;
    }
    
    // Add auth token if available
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Generate request ID and track start time
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const source = createCancellableSource();
    
    // Store request metadata
    config.headers['X-Request-ID'] = requestId;
    config.cancelToken = source.token;
    
    const requestConfig = {
      ...config,
      metadata: { 
        requestId, 
        startTime,
        source,
        requestKey
      }
    };
    
    // Track the request
    metrics.track({
      type: 'api',
      name: 'api.request',
      value: 1,
      metadata: {
        url: config.url,
        method: config.method?.toUpperCase() || 'GET',
        requestId,
        version: API_VERSION
      },
    });
    
    // Store the request promise for deduplication
    if (isIdempotent) {
      const requestPromise = axios({
        ...requestConfig,
        headers: {
          ...requestConfig.headers,
          'X-Request-ID': requestId
        } as any
      }).finally(() => {
        pendingRequests.delete(requestKey);
      });
      pendingRequests.set(requestKey, requestPromise);
      return requestPromise as any;
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
          fromCache: (response.headers as any)['x-cache'] === 'HIT',
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
  async (error) => {
    const originalRequest = error.config as any;
    const { requestId } = originalRequest?.metadata || {};

    // If error is not 401 or we've already retried, reject
    if (error.response?.status !== 401 || originalRequest._retry) {
      return Promise.reject(error);
    }

    // Skip refresh for login/register endpoints
    if (originalRequest.url?.includes('/auth/login') || 
        originalRequest.url?.includes('/auth/register')) {
      return Promise.reject(error);
    }

    originalRequest._retry = true;
    
    try {
      // Try to refresh the token
      const refreshToken = localStorage.getItem('refresh_token');
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const response = await axios.post(
        `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001/api'}/auth/refresh-token`,
        { refreshToken },
        { withCredentials: true }
      );
      
      const { accessToken } = response.data;
      localStorage.setItem('access_token', accessToken);
      
      // Update the authorization header
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
        console.error('Session expired. Please log in again.');
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    // Handle other errors
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
        requestId,
      });
    } else if (error.request) {
      // The request was made but no response was received
      console.error('API Request Error:', {
        request: error.request,
        requestId,
      });
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('API Error:', {
        message: error.message,
        requestId,
      });
    }
    
    return Promise.reject(error);
  }
);

// Cancel all pending requests is now available as api.cancelAllRequests()

export const api = {
  get: async <T = any>(
    url: string, 
    params?: any, 
    config: any = {}
  ): Promise<T> => {
    const response = await apiClient.get(url, { ...config, params });
    return response.data;
  },

  post: async <T = any, D = any>(
    url: string, 
    data?: D, 
    config: any = {}
  ): Promise<T> => {
    const response = await apiClient.post(url, data, config);
    return response.data;
  },

  put: async <T = any, D = any>(
    url: string, 
    data?: D, 
    config: any = {}
  ): Promise<T> => {
    const response = await apiClient.put(url, data, config);
    return response.data;
  },

  delete: async <T = any>(
    url: string, 
    config: any = {}
  ): Promise<T> => {
    const response = await apiClient.delete(url, config);
    return response.data;
  },

  // Batch multiple requests
  batch: <T = any>(requests: Array<() => Promise<T>>): Promise<T[]> => {
    return Promise.all(requests.map(fn => fn()));
  },

  // Get the current API version
  getVersion: (): string => API_VERSION,

  // Set authentication token
  setAuthToken: (token: string | null): void => {
    if (token) {
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete apiClient.defaults.headers.common['Authorization'];
    }
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
  handleError: (error: any): never => {
    if (axios.isCancel(error)) {
      throw new Error('Request was cancelled');
    }
    
    if (error.response) {
      const { status, data } = error.response;
      const message = data?.message || error.message || 'An error occurred';
      
      const errorWithStatus = new Error(message) as any;
      errorWithStatus.status = status;
      errorWithStatus.data = data;
      
      throw errorWithStatus;
    } else if (error.request) {
      throw new Error('Network error - no response received');
    } else {
      throw new Error(error.message || 'Unknown error');
    }
  }
};

// Client instances
export const clients = {
  main: apiClient,
  auth: authApiClient,
  analysis: analysisApiClient
};

// Response creation is handled by Axios internally

// Export the API clients
export { apiClient, authApiClient, analysisApiClient };
export default api;
