import axios from 'axios';
import { pythonConfig } from './python-config';
import { logger } from './logger';
import { getAccessToken } from "../auth/getAccessToken";

// 🔥 FIX 4 — Import AbortController for timeout enforcement
// Use global AbortController available in Node.js 15+
declare global {
  interface AbortSignal {
    readonly aborted: boolean;
    addEventListener(type: string, listener: () => void): void;
    removeEventListener(type: string, listener: () => void): void;
  }
  interface AbortController {
    readonly signal: AbortSignal;
    abort(): void;
  }
  var AbortController: {
    new (): AbortController;
    prototype: AbortController;
  };
}

// Define custom types since axios types are problematic
type CustomAxiosRequestConfig = {
  url?: string;
  method?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  data?: unknown;
  params?: unknown;
  timeout?: number;
  validateStatus?: (status: number) => boolean;
};

type CustomAxiosResponse<T = unknown> = {
  data: T;
  status: number;
  statusText: string;
  headers?: Record<string, string>;
  config?: CustomAxiosRequestConfig;
};

type CustomAxiosInstance = {
  request<T = unknown>(config: CustomAxiosRequestConfig): Promise<CustomAxiosResponse<T>>;
  get<T = unknown>(url: string, config?: CustomAxiosRequestConfig): Promise<CustomAxiosResponse<T>>;
  post<T = unknown>(url: string, data?: unknown, config?: CustomAxiosRequestConfig): Promise<CustomAxiosResponse<T>>;
  put<T = unknown>(url: string, data?: unknown, config?: CustomAxiosRequestConfig): Promise<CustomAxiosResponse<T>>;
  delete<T = unknown>(url: string, config?: CustomAxiosRequestConfig): Promise<CustomAxiosResponse<T>>;
  interceptors: {
    request: {
      use: (onFulfilled: (config: CustomAxiosRequestConfig) => CustomAxiosRequestConfig, onRejected?: (error: unknown) => unknown) => void;
    };
    response: {
      use: (onFulfilled: (response: CustomAxiosResponse<unknown>) => CustomAxiosResponse<unknown>, onRejected?: (error: unknown) => unknown) => void;
    };
  };
};

// Create custom axios instance with proper typing
const createCustomAxios = (): CustomAxiosInstance => {
  const instance = (axios as unknown as {
    create: (config: unknown) => CustomAxiosInstance;
  }).create({
    baseURL: pythonConfig.apiUrl,
    timeout: pythonConfig.requestTimeout,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': pythonConfig.apiKey,
    },
  });

  return {
    request: (config: CustomAxiosRequestConfig) => instance.request(config),
    get: (url: string, config?: CustomAxiosRequestConfig) => instance.get(url, config),
    post: (url: string, data?: unknown, config?: CustomAxiosRequestConfig) => instance.post(url, data, config),
    put: (url: string, data?: unknown, config?: CustomAxiosRequestConfig) => instance.put(url, data, config),
    delete: (url: string, config?: CustomAxiosRequestConfig) => instance.delete(url, config),
    interceptors: instance.interceptors
  } as CustomAxiosInstance;
};

const customAxios = createCustomAxios();

// Circuit breaker states
export enum CircuitState {
  CLOSED = 'CLOSED',   // Normal operation
  OPEN = 'OPEN',       // Circuit is open, fail fast
  HALF_OPEN = 'HALF_OPEN' // Semi-open, testing if service recovered
}

// Circuit breaker configuration
interface CircuitBreakerConfig {
  failureThreshold: number;    // Failures before opening circuit
  resetTimeout: number;        // Time to wait before trying again (ms)
  monitoringPeriod: number;     // Period to monitor failures (ms)
  halfOpenMaxCalls: number;   // Max calls in half-open state
}

interface CircuitBreakerStats {
  failures: number;
  successes: number;
  lastFailureTime: number;
  lastSuccessTime: number;
  state: CircuitState;
}

class CircuitBreaker {
  private config: CircuitBreakerConfig;
  private stats: CircuitBreakerStats;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private lastSuccessTime = 0;

  constructor(config: Partial<CircuitBreakerConfig> = {}) {
    this.config = {
      failureThreshold: 5,
      resetTimeout: 60000,        // 1 minute
      monitoringPeriod: 300000,     // 5 minutes
      halfOpenMaxCalls: 3,
      ...config
    };

    this.stats = {
      failures: 0,
      successes: 0,
      lastFailureTime: 0,
      lastSuccessTime: 0,
      state: CircuitState.CLOSED
    };
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    const now = Date.now();

    // Reset counters if monitoring period has passed
    if (now - this.lastFailureTime > this.config.monitoringPeriod) {
      this.failureCount = 0;
      this.successCount = 0;
    }

    // Check circuit state
    if (this.stats.state === CircuitState.OPEN) {
      if (now - this.lastFailureTime < this.config.resetTimeout) {
        throw new Error('Circuit breaker is OPEN - service temporarily unavailable');
      } else {
        // Try to close circuit (move to half-open)
        this.stats.state = CircuitState.HALF_OPEN;
        logger.info('Circuit breaker moving to HALF-OPEN state');
      }
    }

    if (this.stats.state === CircuitState.HALF_OPEN && this.successCount >= this.config.halfOpenMaxCalls) {
      throw new Error('Circuit breaker HALF-OPEN call limit exceeded');
    }

    try {
      const result = await operation();
      this.onSuccess(now);
      return result;
    } catch (error) {
      this.onFailure(now);
      throw error;
    }
  }

  private onSuccess(now: number): void {
    this.successCount++;
    this.lastSuccessTime = now;
    this.stats.successes++;
    this.stats.lastSuccessTime = now;

    if (this.stats.state === CircuitState.HALF_OPEN) {
      this.stats.state = CircuitState.CLOSED;
      this.successCount = 0;
      this.failureCount = 0;
      logger.info('Circuit breaker moving to CLOSED state');
    }
  }

  private onFailure(now: number): void {
    this.failureCount++;
    this.lastFailureTime = now;
    this.stats.failures++;
    this.stats.lastFailureTime = now;

    if (this.failureCount >= this.config.failureThreshold) {
      this.stats.state = CircuitState.OPEN;
      logger.warn(`Circuit breaker moving to OPEN state - ${this.failureCount} failures detected`);
    }
  }

  getStats(): CircuitBreakerStats {
    return { ...this.stats };
  }

  getState(): CircuitState {
    return this.stats.state;
  }

  reset(): void {
    this.failureCount = 0;
    this.successCount = 0;
    this.stats = {
      failures: 0,
      successes: 0,
      lastFailureTime: 0,
      lastSuccessTime: 0,
      state: CircuitState.CLOSED
    };
    logger.info('Circuit breaker reset to CLOSED state');
  }
}

// Health check interface
interface HealthCheckResult {
  healthy: boolean;
  responseTime: number;
  error?: string;
  timestamp: number;
}

class PythonServiceHealth {
  private lastHealthCheck = 0;
  private healthCheckInterval = 30000; // 30 seconds
  private isHealthy = false;

  async checkHealth(): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    try {
      const response = await customAxios.get(
        `${pythonConfig.apiUrl}/health`,
        {
          timeout: 5000, // 5 second timeout for health check
          // Health endpoint should be public - no API key required for Python service
        }
      );

      const responseTime = Date.now() - startTime;
      const healthy = response.status === 200 && (response.data as { status?: string })?.status === 'healthy';

      this.isHealthy = healthy;
      this.lastHealthCheck = Date.now();

      return {
        healthy,
        responseTime,
        timestamp: this.lastHealthCheck
      };
    } catch (error) {
      this.isHealthy = false;
      this.lastHealthCheck = Date.now();
      
      return {
        healthy: false,
        responseTime: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: this.lastHealthCheck
      };
    }
  }

  shouldCheckHealth(): boolean {
    return Date.now() - this.lastHealthCheck > this.healthCheckInterval;
  }

  isServiceHealthy(): boolean {
    return this.isHealthy;
  }
}

// Create singleton instances
export const circuitBreaker = new CircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 60000,  // 1 minute
  monitoringPeriod: 120000, // 2 minutes
  halfOpenMaxCalls: 2
});

export const healthChecker = new PythonServiceHealth();

// Enhanced Python bridge with circuit breaker and health checks
export class EnhancedPythonBridge {
  private retryConfig = {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    backoffFactor: 2
  };

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private calculateRetryDelay(attempt: number): number {
    const delay = this.retryConfig.baseDelay * Math.pow(this.retryConfig.backoffFactor, attempt - 1);
    const jitter = Math.random() * 0.1 * delay; // Add 10% jitter
    return Math.min(delay + jitter, this.retryConfig.maxDelay);
  }

  async request<T>(config: {
    method: 'GET' | 'POST' | 'PUT' | 'DELETE';
    url: string;
    data?: unknown;
    headers?: Record<string, string>;
    timeout?: number;
    userToken?: string; // Pass user token from request context
    signal?: AbortSignal; // Add AbortSignal support
  }): Promise<T> {
    // 🔥 PRODUCTION: Reduced timeout for stability
    const MAX_JOB_TIME = 2 * 60 * 1000; // 2 minutes hard cap
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      logger.warn('[PYTHON BRIDGE] Hard timeout reached, aborting request', {
        url: config.url,
        method: config.method,
        maxTime: MAX_JOB_TIME
      });
      controller.abort();
    }, MAX_JOB_TIME);

    // Combine external signal with our timeout signal manually
    let combinedSignal = controller.signal;
    let shouldAbort = false;

    // Listen to external signal if provided
    if (config.signal) {
      const checkExternalSignal = () => {
        if (config.signal!.aborted && !shouldAbort) {
          shouldAbort = true;
          controller.abort();
        }
      };
      
      config.signal.addEventListener('abort', checkExternalSignal);
      
      // Clean up listener when either signal aborts
      const cleanup = () => {
        config.signal!.removeEventListener('abort', checkExternalSignal);
      };
      
      combinedSignal.addEventListener('abort', cleanup);
    }

    try {
      const result = await this.executeRequestWithTimeout<T>({
        ...config,
        signal: combinedSignal
      });
      
      clearTimeout(timeout);
      return result;
    } catch (error) {
      clearTimeout(timeout);
      
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Python bridge timeout exceeded - analysis took too long');
      }
      
      throw error;
    }
  }

  private async executeRequestWithTimeout<T>(config: {
    method: 'GET' | 'POST' | 'PUT' | 'DELETE';
    url: string;
    data?: unknown;
    headers?: Record<string, string>;
    timeout?: number;
    userToken?: string;
    signal: AbortSignal;
  }): Promise<T> {
    // Perform health check if needed
    if (healthChecker.shouldCheckHealth()) {
      const health = await healthChecker.checkHealth();
      logger.info('Python service health check:', {
        healthy: health.healthy,
        responseTime: health.responseTime,
        error: health.error
      });
    }

    // Execute request through circuit breaker with retry logic
    return circuitBreaker.execute(async (): Promise<T> => {
      let lastError: unknown;
      
      for (let attempt = 1; attempt <= this.retryConfig.maxRetries; attempt++) {
        const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const requestStartTime = Date.now();
        
        try {
          const response = await customAxios.request({
            ...config,
            url: `${pythonConfig.apiUrl}${config.url}`,
            data: config.data,
            headers: {
              'X-API-Key': pythonConfig.apiKey,
              'X-Request-Id': requestId,
              'X-Correlation-Id': requestId,
              // Pass user's JWT token for Python service authentication
              'Authorization': config.userToken ? `Bearer ${config.userToken}` : `Bearer ${await getAccessToken()}`,
              ...config.headers
            },
            timeout: config.timeout || pythonConfig.requestTimeout
          });

          // Runtime assertion: Fail fast on 404 errors
          if (response.status === 404) {
            const isLegacyRoute = config.url.includes('/analyze/') && !config.url.includes('/api/v2/analysis/unified/');
            if (isLegacyRoute) {
              logger.warn(`[DEPRECATED] Legacy Python route used: ${config.url}. Please migrate to /api/v2/analysis/unified/* routes`);
            }
            throw new Error(`Python endpoint not found: ${config.url}. This indicates a route mismatch between Node and Python services.`);
          }

          // Check for service unavailability
          if (response.status === 503 || response.status === 504) {
            logger.warn(`Python service unavailable: ${config.url}`, {
              status: response.status,
              circuitState: circuitBreaker.getState()
            });
            throw new Error(`ML service temporarily unavailable. Please try again later.`);
          }

          logger.info('Python bridge request completed', {
            requestId,
            method: config.method,
            url: config.url,
            status: response.status,
            responseTime: Date.now() - requestStartTime,
            circuitState: circuitBreaker.getState(),
            attempt: attempt
          });

          return response.data as T;
          
        } catch (error) {
          lastError = error;
          
          // Don't retry on certain errors
          if (error instanceof Error && (
            error.message.includes('404') || 
            error.message.includes('400') ||
            error.message.includes('401') ||
            error.message.includes('403')
          )) {
            throw error;
          }
          
          // Retry on transient errors
          if (attempt < this.retryConfig.maxRetries) {
            const delay = this.calculateRetryDelay(attempt);
            logger.warn(`Python bridge request failed, retrying in ${delay}ms`, {
              requestId,
              attempt: attempt,
              maxRetries: this.retryConfig.maxRetries,
              error: error instanceof Error ? error.message : 'Unknown error'
            });
            
            await this.delay(delay);
            continue;
          }
          
          // Final attempt failed
          logger.error('Python bridge request failed after all retries', {
            requestId,
            attempts: attempt,
            error: error instanceof Error ? error.message : 'Unknown error'
          });
          
          throw error;
        }
      }
      
      throw lastError; // Should never reach here
    });
  }

  getCircuitStats(): CircuitBreakerStats {
    return circuitBreaker.getStats();
  }

  async getHealthStatus(): Promise<HealthCheckResult> {
    return healthChecker.checkHealth();
  }

  isHealthy(): boolean {
    return healthChecker.isServiceHealthy();
  }
}

export const enhancedPythonBridge = new EnhancedPythonBridge();
