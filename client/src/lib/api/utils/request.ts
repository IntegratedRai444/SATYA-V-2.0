import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { v4 as uuidv4 } from 'uuid';
import API_CONFIG from '../config';
import {
  ApiError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  ServerError,
  UnauthorizedError,
  ForbiddenError,
  NotFoundError,
  ValidationError,
  BadRequestError
} from '../errors';

type RequestOptions = AxiosRequestConfig & {
  id?: string;
  retryCount?: number;
  skipAuth?: boolean;
  cacheTtl?: number;
  metadata?: {
    requestId: string;
    startTime: number;
    cacheKey?: string;
    fromCache?: boolean;
  };
};

const pendingRequests = new Map<string, Promise<AxiosResponse>>();
const requestCache = new Map<string, { data: any; timestamp: number; expiry: number }>();

export const generateRequestId = (): string => uuidv4();

export const createRequestConfig = (config: RequestOptions = {}): RequestOptions => {
  const requestId = config.id || generateRequestId();
  const headers = {
    ...API_CONFIG.DEFAULT_HEADERS,
    'X-Request-Id': requestId,
    ...config.headers,
  };

  return {
    ...config,
    headers,
    timeout: config.timeout || API_CONFIG.TIMEOUT,
    withCredentials: config.withCredentials ?? API_CONFIG.CORS.credentials === 'include',
    metadata: {
      ...config.metadata,
      requestId,
      startTime: Date.now(),
    },
  };
};

export const handleRequestError = (error: AxiosError): never => {
  if (axios.isCancel(error)) {
    throw new ApiError('Request was cancelled', 'CANCELLED', 0, error.config);
  }

  if (!error.response) {
    if (error.code === 'ECONNABORTED') {
      throw new TimeoutError(API_CONFIG.ERROR_MESSAGES.TIMEOUT, error.config);
    }
    throw new NetworkError(API_CONFIG.ERROR_MESSAGES.NETWORK, error.config);
  }

  const { status, data, config } = error.response;
  const message = (data && typeof data === 'object' && 'message' in data && typeof data.message === 'string') 
    ? data.message 
    : error.message || API_CONFIG.ERROR_MESSAGES.UNKNOWN;

  switch (status) {
    case 400:
      if (data && typeof data === 'object' && 'errors' in data) {
        throw new ValidationError(API_CONFIG.ERROR_MESSAGES.VALIDATION, data.errors as Record<string, string[]>, data, config);
      }
      throw new BadRequestError(message, data, config);
    case 401:
      throw new UnauthorizedError(API_CONFIG.ERROR_MESSAGES.UNAUTHORIZED, config);
    case 403:
      throw new ForbiddenError(API_CONFIG.ERROR_MESSAGES.FORBIDDEN, config);
    case 404:
      throw new NotFoundError(API_CONFIG.ERROR_MESSAGES.NOT_FOUND, config);
    case 408:
      throw new TimeoutError(API_CONFIG.ERROR_MESSAGES.TIMEOUT, config);
    case 429: {
      const retryAfter = parseInt(
        error.response?.headers?.['retry-after'] || '0',
        10
      );
      throw new RateLimitError(
        'Too many requests',
        config,
        retryAfter * 1000 || undefined
      );
    }
    case 500:
    case 502:
    case 503:
    case 504:
      throw new ServerError(API_CONFIG.ERROR_MESSAGES.SERVER, data, config);
    default:
      throw new ApiError(message, 'API_ERROR', status, config);
  }
};

export const getFromCache = (key: string) => {
  const cached = requestCache.get(key);
  if (!cached) return null;
  
  if (Date.now() > cached.expiry) {
    requestCache.delete(key);
    return null;
  }
  
  return cached.data;
};

export const setToCache = (key: string, data: any, ttl: number = API_CONFIG.DEFAULT_CACHE_TTL) => {
  requestCache.set(key, {
    data,
    timestamp: Date.now(),
    expiry: Date.now() + ttl,
  });
};

export const clearCache = () => {
  requestCache.clear();
};

export const getCacheStats = () => ({
  size: requestCache.size,
  keys: Array.from(requestCache.keys()),
});

export const createCancellableSource = () => {
  const source = axios.CancelToken.source();
  return {
    token: source.token,
    cancel: (message?: string) => source.cancel(message || 'Request cancelled by user'),
  };
};

export const cancelRequest = (requestId: string, message: string = 'Request cancelled'): boolean => {
  const pendingRequest = pendingRequests.get(requestId);
  if (pendingRequest) {
    const source = createCancellableSource();
    source.cancel(message);
    pendingRequests.delete(requestId);
    return true;
  }
  return false;
};

export const cancelAllRequests = (message: string = 'All requests cancelled'): void => {
  pendingRequests.forEach((_, requestId) => {
    cancelRequest(requestId, message);
  });
};

export const isRetryableError = (error: any): boolean => {
  if (!error || !error.response) return false;
  
  const { status } = error.response;
  return [
    408, // Request Timeout
    429, // Too Many Requests
    500, // Internal Server Error
    502, // Bad Gateway
    503, // Service Unavailable
    504, // Gateway Timeout
  ].includes(status);
};

export const getRetryDelay = (retryCount: number): number => {
  const delay = Math.min(
    API_CONFIG.RETRY_DELAY * Math.pow(2, retryCount - 1) + Math.random() * 1000, // Add jitter
    API_CONFIG.MAX_RETRY_DELAY
  );
  return delay;
};
