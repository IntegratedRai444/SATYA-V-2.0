import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import { getCsrfToken } from './csrfService';

// Request cache for deduplication
interface PendingRequest {
  promise: Promise<unknown>;
  timestamp: number;
  abortController: AbortController;
}

class RequestDeduplicator {
  private pendingRequests = new Map<string, PendingRequest>();
  private readonly CACHE_TTL = 30000; // 30 seconds

  private generateKey(method: string, url: string, data?: unknown): string {
    const dataHash = data ? JSON.stringify(data) : '';
    return `${method}:${url}:${dataHash}`;
  }

  async deduplicate<T>(
    method: string,
    url: string,
    requestFn: (signal: AbortSignal) => Promise<T>,
    data?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const key = this.generateKey(method, url, data);
    
    // Check if we have a pending request
    const existing = this.pendingRequests.get(key);
    if (existing && Date.now() - existing.timestamp < this.CACHE_TTL) {
      // Return existing promise if within cache TTL
      return existing.promise as Promise<T>;
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    
    // Create the request promise
    const promise = requestFn(abortController.signal)
      .finally(() => {
        // Clean up after request completes
        this.pendingRequests.delete(key);
      });

    // Store the pending request
    this.pendingRequests.set(key, {
      promise,
      timestamp: Date.now(),
      abortController
    });

    // If request has a timeout, set up abort
    if (options?.timeout) {
      setTimeout(() => {
        if (!abortController.signal.aborted) {
          abortController.abort();
        }
      }, options.timeout);
    }

    return promise;
  }

  cancelRequest(method: string, url: string, data?: unknown): boolean {
    const key = this.generateKey(method, url, data);
    const pending = this.pendingRequests.get(key);
    
    if (pending) {
      pending.abortController.abort();
      this.pendingRequests.delete(key);
      return true;
    }
    
    return false;
  }

  clear(): void {
    // Cancel all pending requests
    for (const [, request] of this.pendingRequests) {
      request.abortController.abort();
    }
    this.pendingRequests.clear();
  }
}

// Retry mechanism with exponential backoff
class RetryManager {
  private readonly DEFAULT_MAX_RETRIES = 3;
  private readonly DEFAULT_BASE_DELAY = 1000; // 1 second
  private readonly DEFAULT_MAX_DELAY = 30000; // 30 seconds
  private readonly DEFAULT_BACKOFF_FACTOR = 2;

  async executeWithRetry<T>(
    requestFn: () => Promise<T>,
    options: {
      maxRetries?: number;
      baseDelay?: number;
      maxDelay?: number;
      backoffFactor?: number;
      retryCondition?: (error: unknown) => boolean;
      onRetry?: (error: unknown, attempt: number) => void;
    } = {}
  ): Promise<T> {
    const {
      maxRetries = this.DEFAULT_MAX_RETRIES,
      baseDelay = this.DEFAULT_BASE_DELAY,
      maxDelay = this.DEFAULT_MAX_DELAY,
      backoffFactor = this.DEFAULT_BACKOFF_FACTOR,
      retryCondition = this.defaultRetryCondition,
      onRetry
    } = options;

    let lastError: unknown;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error;
        
        // Don't retry on the last attempt
        if (attempt === maxRetries) {
          throw error;
        }
        
        // Check if we should retry this error
        if (!retryCondition(error)) {
          throw error;
        }
        
        // Calculate delay with exponential backoff and jitter
        const delay = Math.min(
          baseDelay * Math.pow(backoffFactor, attempt),
          maxDelay
        );
        
        // Add jitter to prevent thundering herd
        const jitter = delay * 0.1 * Math.random();
        const finalDelay = delay + jitter;
        
        // Call retry callback if provided
        if (onRetry) {
          onRetry(error, attempt + 1);
        }
        
        // Wait before retrying
        await this.sleep(finalDelay);
      }
    }
    
    throw lastError;
  }

  private defaultRetryCondition(error: unknown): boolean {
    // Retry on network errors and 5xx server errors
    if (!error || typeof error !== 'object' || !('response' in error)) {
      // Network error or invalid error object
      return true;
    }
    
    const errorWithResponse = error as { response?: { status?: number } };
    const status = errorWithResponse.response?.status;
    return typeof status === 'number' && (status >= 500 || status === 429); // Retry on server errors and rate limiting
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

import { API_CONFIG } from '../../config/urls';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
});

export type RequestHeaders = {
  'X-Request-Id'?: string;
  'Content-Type'?: string;
  'X-CSRF-Token'?: string;
  [key: string]: string | undefined;
};

export type RequestOptions = {
  id?: string;
  retryCount?: number;
  timeout?: number;
  skipAuth?: boolean;
  signal?: AbortSignal;
  headers?: RequestHeaders;
  withCredentials?: boolean;
  skipCSRF?: boolean;
  [key: string]: unknown; // For any additional properties
};

export class BaseService {
  protected basePath: string;
  protected client: typeof apiClient;
  protected isAuthService: boolean;
  private deduplicator = new RequestDeduplicator();
  private retryManager = new RetryManager();

  constructor(basePath: string, isAuthService: boolean = false) {
    this.basePath = basePath;
    this.isAuthService = isAuthService;
    this.client = apiClient;
  }

  protected async get<T = unknown>(
    endpoint: string = '',
    params: Record<string, unknown> = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.basePath}${endpoint}`;
    const requestFn = async (signal: AbortSignal) => {
      const headers: RequestHeaders = {
        ...options.headers,
        'X-Request-Id': options.id || uuidv4(),
      };

      // Add CSRF token for non-GET requests or when explicitly requested
      if (!options.skipCSRF && !options.skipAuth) {
        const csrfToken = await getCsrfToken();
        if (csrfToken) {
          headers['X-CSRF-Token'] = csrfToken;
        }
      }

      const response = await this.client.get<T>(
        url,
        {
          ...options,
          params,
          headers,
          withCredentials: options.withCredentials ?? true,
          signal
        }
      );
      return response.data;
    };

    return this.deduplicator.deduplicate('GET', url, requestFn, undefined, options);
  }

  protected async post<T = unknown>(
    endpoint: string = '',
    data: unknown = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.basePath}${endpoint}`;
    const requestFn = async (signal: AbortSignal) => {
      return this.retryManager.executeWithRetry(
        async () => {
          const headers = {
            ...options.headers,
            'X-Request-Id': options.id || uuidv4(),
            'Content-Type': 'application/json',
          };

          // Add CSRF token for non-GET requests or when explicitly requested
          if (!options.skipCSRF && !options.skipAuth) {
            const csrfToken = await getCsrfToken();
            if (csrfToken) {
              headers['X-CSRF-Token'] = csrfToken;
            }
          }

          const response = await this.client.post<T>(
            url,
            data,
            {
              ...options,
              headers,
              withCredentials: options.withCredentials ?? true,
              signal
            }
          );
          return response.data;
        },
        {
          maxRetries: 3,
          baseDelay: 1000,
          onRetry: (error, attempt) => {
            console.warn(`Retrying POST request to ${url}, attempt ${attempt}:`, error);
          }
        }
      );
    };

    return this.deduplicator.deduplicate('POST', url, requestFn, data, options);
  }

  protected async put<T = unknown>(
    endpoint: string = '',
    data: unknown = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.basePath}${endpoint}`;
    const requestFn = async (signal: AbortSignal) => {
      return this.retryManager.executeWithRetry(
        async () => {
          const headers = {
            ...options.headers,
            'X-Request-Id': options.id || uuidv4(),
            'Content-Type': 'application/json',
          };

          // Add CSRF token for non-GET requests or when explicitly requested
          if (!options.skipCSRF && !options.skipAuth) {
            const csrfToken = await getCsrfToken();
            if (csrfToken) {
              headers['X-CSRF-Token'] = csrfToken;
            }
          }

          const response = await this.client.put<T>(
            url,
            data,
            {
              ...options,
              headers,
              withCredentials: options.withCredentials ?? true,
              signal
            }
          );
          return response.data;
        },
        {
          maxRetries: 3,
          baseDelay: 1000,
          onRetry: (error, attempt) => {
            console.warn(`Retrying PUT request to ${url}, attempt ${attempt}:`, error);
          }
        }
      );
    };

    return this.deduplicator.deduplicate('PUT', url, requestFn, data, options);
  }

  protected async delete<T = unknown>(
    endpoint: string = '',
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.basePath}${endpoint}`;
    const requestFn = async (signal: AbortSignal) => {
      return this.retryManager.executeWithRetry(
        async () => {
          const headers = {
            ...options.headers,
            'X-Request-Id': options.id || uuidv4(),
          };

          // Add CSRF token for non-GET requests or when explicitly requested
          if (!options.skipCSRF && !options.skipAuth) {
            const csrfToken = await getCsrfToken();
            if (csrfToken) {
              headers['X-CSRF-Token'] = csrfToken;
            }
          }

          const response = await this.client.delete<T>(
            url,
            {
              ...options,
              headers,
              withCredentials: options.withCredentials ?? true,
              signal
            }
          );
          return response.data;
        },
        {
          maxRetries: 2, // Fewer retries for DELETE
          baseDelay: 1000,
          onRetry: (error, attempt) => {
            console.warn(`Retrying DELETE request to ${url}, attempt ${attempt}:`, error);
          }
        }
      );
    };

    return this.deduplicator.deduplicate('DELETE', url, requestFn, undefined, options);
  }

  protected async patch<T = unknown>(
    endpoint: string = '',
    data: unknown = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.basePath}${endpoint}`;
    const requestFn = async (signal: AbortSignal) => {
      return this.retryManager.executeWithRetry(
        async () => {
          const response = await this.client.patch<T>(
            url,
            data,
            {
              ...options,
              headers: {
                ...options.headers,
                'X-Request-Id': options.id || uuidv4(),
              },
              signal
            }
          );
          return response.data;
        },
        {
          maxRetries: 2,
          baseDelay: 1000,
          onRetry: (error, attempt) => {
            console.warn(`Retrying PATCH request to ${url}, attempt ${attempt}:`, error);
          }
        }
      );
    };

    return this.deduplicator.deduplicate('PATCH', url, requestFn, data, options);
  }

  // Public methods to cancel requests
  public cancelRequest(method: string, endpoint: string, data?: unknown): boolean {
    const url = `${this.basePath}${endpoint}`;
    return this.deduplicator.cancelRequest(method, url, data);
  }

  public cancelAllRequests(): void {
    this.deduplicator.clear();
  }

  protected createAbortController(): AbortController {
    return new AbortController();
  }
  protected generateRequestId(): string {
    return uuidv4();
  }
}
