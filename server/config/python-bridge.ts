import axios from 'axios';
import { pythonConfig } from './python-config';
import { logger } from './logger';

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
          // Health endpoint is public - no API key required
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
  async request<T>(config: {
    method: 'GET' | 'POST' | 'PUT' | 'DELETE';
    url: string;
    data?: unknown;
    headers?: Record<string, string>;
    timeout?: number;
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

    // Execute request through circuit breaker
    return circuitBreaker.execute(async (): Promise<T> => {
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const requestStartTime = Date.now();
      
      const response = await customAxios.request({
        ...config,
        url: `${pythonConfig.apiUrl}${config.url}`,
        data: config.data,
        headers: {
          'X-API-Key': pythonConfig.apiKey,
          'X-Request-Id': requestId,
          'X-Correlation-Id': requestId,
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
        circuitState: circuitBreaker.getState()
      });

      return response.data as T;
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
