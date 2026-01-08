import axios, { AxiosInstance, AxiosResponse, AxiosError, AxiosRequestConfig, InternalAxiosRequestConfig, AxiosHeaders } from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { metrics, trackError as logError } from '@/lib/services/metrics';

// Types
type CacheEntry = {
  timestamp: number;
  response: AxiosResponse;
  expiry: number;
};

type RetryConfig = {
  retries: number;
  retryDelay: number;
  retryOn: number[];
  maxRetryDelay: number;
};

// Extend InternalAxiosRequestConfig to include our custom properties
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
  const { method, url, params, data, paramsSerializer } = config;
  const serializedParams = paramsSerializer ? paramsSerializer(params) : JSON.stringify(params);
  const serializedData = typeof data === 'string' ? data : JSON.stringify(data);
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
    (config: RequestConfig) => {
      const requestId = config.headers?.['X-Request-Id'] as string || uuidv4();
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
      const { config, status, statusText, data } = response;
      const requestConfig = config as RequestConfig;
      const { requestId, cacheKey, startTime } = requestConfig.metadata || {};
      
      // Log successful request
      const duration = startTime ? Date.now() - startTime : 0;
      console.log(`[${requestId}] ${requestConfig.method?.toUpperCase()} ${requestConfig.url} ${status} (${duration}ms)`);
      
      // Cache successful GET responses
      if (requestConfig.method?.toUpperCase() === 'GET' && status === 200 && cacheKey) {
        setCachedResponse(cacheKey, response, requestConfig.cacheTtl);
      }
      
      // Remove from pending requests
      if (requestConfig.id) {
        pendingRequests.delete(requestConfig.id);
      }
      
      return response;
    },
    async (error) => {
      const { config, response } = error;
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
    const batchResponses = await Promise.all(
      currentBatch.map(({ config }) => 
        apiClient.request(config)
          .then(res => ({ success: true, data: res.data }))
          .catch(error => ({ success: false, error }))
      )
    );
    
    // Resolve or reject each promise in the batch
    currentBatch.forEach(({ resolve, reject }, index) => {
      const response = batchResponses[index];
      if (response.success) {
        resolve(response.data);
      } else {
        reject(response.error);
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
          resolve: (data) => resolve({ ...config, data, status: 200, statusText: 'OK', headers: {}, config }),
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
      const requestPromise = axios(requestConfig)
        .finally(() => {
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

// Response interceptor for handling errors and tracking responses
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    const { requestId, startTime, requestKey } = (response.config as any).metadata || {};
    
    if (requestId && startTime) {
      const duration = Date.now() - startTime;
      
      // Track successful response
      metrics.track({
        type: 'api',
        name: 'api.response',
        value: duration,
        metadata: {
          url: response.config.url,
          method: response.config.method?.toUpperCase(),
          status: response.status,
          statusText: response.statusText,
          requestId,
          duration,
          version: API_VERSION,
          fromCache: response.headers['x-cache'] === 'HIT'
        },
      });
      
      // Clean up request cache
      if (requestKey) {
        pendingRequests.delete(requestKey);
      }
    }
    
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as any;
    const { requestId, startTime, requestKey } = originalRequest?.metadata || {};
    
    // Track the error
    if (requestId && startTime) {
      // Remove unused duration variable
      const status = error.response?.status || 0;
      
logError(
        new Error(`API Error: ${error.message} (${status} ${error.response?.statusText})`),
        'api_error',
        { status, url: originalRequest?.url }
      );
      
      // Clean up request cache on error
      if (requestKey) {
        pendingRequests.delete(requestKey);
      }
    }
    
    // Handle request cancellation
    if (axios.isCancel(error)) {
      return Promise.reject({
        code: 'REQUEST_CANCELLED',
        message: 'Request was cancelled',
        isCancelled: true
      });
    }
    
    // Handle 401 Unauthorized (token refresh)
    if (error.response?.status === 401 && !originalRequest?._retry) {
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
        }
        
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


// Helper function to cancel all pending requests
const cancelAllRequests = (reason = 'User navigated away') => {
  pendingRequests.forEach((_, key) => {
    const request = pendingRequests.get(key) as any;
    if (request?.cancel) {
      request.cancel(reason);
export const api = {
  get: <T = any>(
    url: string, 
    params?: any, 
    config: RequestConfig = {}
  ): Promise<T> => {
    return apiClient.get<T>(url, { ...config, params })
      .then(response => response.data)
      .catch(handleApiError);
  },
  
  post: <T = any, D = any>(
    url: string, 
    data?: D, 
    config: RequestConfig = {}
  ): Promise<T> => {
    return apiClient.post<T>(url, data, config)
      .then(response => response.data)
      .catch(handleApiError);
  },
  
  put: <T = any, D = any>(
    url: string, 
    data?: D, 
    config: RequestConfig = {}
  ): Promise<T> => {
    return apiClient.put<T>(url, data, config)
      .then(response => response.data)
      .catch(handleApiError);
  },
  
  delete: <T = any>(
    url: string, 
    config: RequestConfig = {}
  ): Promise<T> => {
    return apiClient.delete<T>(url, config)
      .then(response => response.data || {} as T)
      .catch(handleApiError);
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
      authApiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      analysisApiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete apiClient.defaults.headers.common['Authorization'];
      delete authApiClient.defaults.headers.common['Authorization'];
      delete analysisApiClient.defaults.headers.common['Authorization'];
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
  getCacheStats: () => {
    return {
      size: requestCache.size,
      keys: Array.from(requestCache.keys())
    };
  }
};

// Error handling utility
const handleApiError = (error: any): never => {
  if (axios.isCancel(error)) {
    throw new ApiError('Request was cancelled', 'CANCELLED', 0, error);
  }
  
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    const { status, data, config } = error.response;
    const message = data?.message || error.message || 'An error occurred';
    
    switch (status) {
      case 400:
        throw new BadRequestError(message, data, config);
      case 401:
        throw new UnauthorizedError(message, config);
      case 403:
        throw new ForbiddenError(message, config);
      case 404:
        throw new NotFoundError(message, config);
      case 408:
        throw new TimeoutError('Request timed out', config);
      case 429:
        throw new RateLimitError('Too many requests', config);
      case 500:
        throw new ServerError('Internal server error', data, config);
      default:
        throw new ApiError(message, 'API_ERROR', status, config);
    }
  } else if (error.request) {
    // The request was made but no response was received
    throw new NetworkError('Network error - no response received', error.config);
  } else {
    // Something happened in setting up the request that triggered an Error
    throw new ApiError(error.message || 'Unknown error', 'UNKNOWN', 0, error.config);
  }
  api as default,
  apiClient,
  authApiClient,
  analysisApiClient,
  cancelAllRequests 
};
