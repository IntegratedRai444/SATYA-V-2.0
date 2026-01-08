import { EventEmitter } from 'events';
import { logger } from '../config/logger';

/**
 * Error types that can trigger the circuit breaker
 */
export enum CircuitBreakerErrorType {
  TIMEOUT = 'TIMEOUT',
  CONNECTION = 'CONNECTION',
  THROTTLED = 'THROTTLED',
  SERVER = 'SERVER',
  CLIENT = 'CLIENT',
  UNKNOWN = 'UNKNOWN'
}

export interface CircuitBreakerError extends Error {
  type: CircuitBreakerErrorType;
  code?: string | number;
  isRetryable?: boolean;
}

export interface CircuitBreakerMetrics {
  failures: number;
  successes: number;
  stateChanges: number;
  lastFailure?: {
    timestamp: number;
    error: CircuitBreakerError;
  };
}

export interface CircuitBreakerOptions {
  /** Number of failures before opening the circuit */
  failureThreshold: number;
  /** Number of successful executions in HALF_OPEN state to close the circuit */
  successThreshold: number;
  /** Time in ms to wait before attempting to close the circuit */
  timeout: number;
  /** Maximum time in ms to wait before timing out the operation */
  operationTimeout?: number;
  /** Whether to enable metrics collection */
  enableMetrics?: boolean;
  /** Custom error filter function to determine if error should be counted as failure */
  errorFilter?: (error: Error) => boolean;
  /** Callback when circuit state changes */
  onStateChange?: (state: 'OPEN' | 'CLOSED' | 'HALF_OPEN') => void;
}

type CircuitState = 'OPEN' | 'CLOSED' | 'HALF_OPEN';

export class CircuitBreaker extends EventEmitter {
  private state: CircuitState = 'CLOSED';
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime: number | null = null;
  private metrics: CircuitBreakerMetrics = {
    failures: 0,
    successes: 0,
    stateChanges: 0
  };
  private halfOpenTimer: NodeJS.Timeout | null = null;
  private readonly defaultOptions: Required<Omit<CircuitBreakerOptions, 'onStateChange'>> = {
    failureThreshold: 5,
    successThreshold: 3,
    timeout: 10000,
    operationTimeout: 5000,
    enableMetrics: true,
    errorFilter: () => true
  };
  private readonly options: Required<Omit<CircuitBreakerOptions, 'onStateChange'>> & Pick<CircuitBreakerOptions, 'onStateChange'>;

  constructor(options?: Partial<CircuitBreakerOptions>) {
    super();
    this.options = { ...this.defaultOptions, ...options };
  }

  /**
   * Execute a function with circuit breaker protection
   * @param fn The async function to execute
   * @param options Override circuit breaker options for this execution
   * @returns The result of the function
   * @throws CircuitBreakerError if circuit is open or execution fails
   */
  async execute<T>(
    fn: () => Promise<T>,
    options?: Partial<Pick<CircuitBreakerOptions, 'operationTimeout' | 'errorFilter'>>
  ): Promise<T> {
    const execOptions = { ...this.options, ...options };
    
    if (this.state === 'OPEN') {
      const now = Date.now();
      const timeSinceLastFailure = this.lastFailureTime ? now - this.lastFailureTime : Infinity;
      
      if (timeSinceLastFailure > this.options.timeout) {
        this.setState('HALF_OPEN');
      } else {
        throw this.createError(
          'Circuit breaker is open',
          CircuitBreakerErrorType.CONNECTION,
          'ECIRCUITOPEN',
          false
        );
      }
    }

    try {
      const result = await this.withTimeout(fn, execOptions.operationTimeout);
      this.recordSuccess();
      return result;
    } catch (error) {
      const shouldCountAsFailure = execOptions.errorFilter?.(error as Error) ?? true;
      if (shouldCountAsFailure) {
        this.recordFailure(error as Error);
      }
      throw error;
    }
  }

  /**
   * Get the current state of the circuit breaker
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get metrics about circuit breaker operation
   */
  getMetrics(): Readonly<CircuitBreakerMetrics> {
    return { ...this.metrics };
  }

  /**
   * Reset the circuit breaker to CLOSED state and clear metrics
   */
  reset(): void {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = null;
    this.metrics = {
      failures: 0,
      successes: 0,
      stateChanges: 0
    };
    if (this.halfOpenTimer) {
      clearTimeout(this.halfOpenTimer);
      this.halfOpenTimer = null;
    }
    this.emit('reset');
  }

  private setState(newState: CircuitState): void {
    if (this.state === newState) return;
    
    const previousState = this.state;
    this.state = newState;
    this.metrics.stateChanges++;
    
    logger.info(`Circuit breaker state changed: ${previousState} -> ${newState}`, {
      previousState,
      newState,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime
    });

    this.emit('stateChange', { from: previousState, to: newState });
    this.options.onStateChange?.(newState);

    if (newState === 'HALF_OPEN' && this.halfOpenTimer === null) {
      // Set a timer to close the circuit if no activity
      this.halfOpenTimer = setTimeout(() => {
        if (this.state === 'HALF_OPEN') {
          this.setState('OPEN');
        }
      }, this.options.timeout);
    } else if (newState !== 'HALF_OPEN' && this.halfOpenTimer) {
      clearTimeout(this.halfOpenTimer);
      this.halfOpenTimer = null;
    }
  }

  private recordSuccess(): void {
    this.metrics.successes++;
    this.failureCount = 0;
    
    if (this.state === 'HALF_OPEN') {
      this.successCount++;
      if (this.successCount >= this.options.successThreshold) {
        this.setState('CLOSED');
        this.successCount = 0;
      }
    }
  }

  private recordFailure(error: Error): void {
    this.metrics.failures++;
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.metrics.lastFailure = {
      timestamp: this.lastFailureTime,
      error: this.normalizeError(error)
    };
    
    if (this.failureCount >= this.options.failureThreshold) {
      this.setState('OPEN');
    }
    
    this.emit('failure', error, this.failureCount);
  }

  private async withTimeout<T>(
    promise: () => Promise<T>,
    timeoutMs: number
  ): Promise<T> {
    let timeout: NodeJS.Timeout;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        reject(this.createError(
          'Operation timed out',
          CircuitBreakerErrorType.TIMEOUT,
          'ETIMEDOUT',
          true
        ));
      }, timeoutMs);
    });

    try {
      return await Promise.race([
        promise(),
        timeoutPromise
      ]);
    } finally {
      clearTimeout(timeout!);
    }
  }

  private normalizeError(error: Error): CircuitBreakerError {
    if (this.isCircuitBreakerError(error)) {
      return error;
    }
    
    const code = (error as any).code || 'EUNKNOWN';
    let type = CircuitBreakerErrorType.UNKNOWN;
    let isRetryable = true;
    
    // Classify error type
    if (code === 'ETIMEDOUT' || error.name === 'TimeoutError') {
      type = CircuitBreakerErrorType.TIMEOUT;
    } else if (code === 'ECONNREFUSED' || code === 'ENOTFOUND') {
      type = CircuitBreakerErrorType.CONNECTION;
    } else if (code === 'ETOOMANYREQUESTS' || code === 429) {
      type = CircuitBreakerErrorType.THROTTLED;
      isRetryable = false;
    } else if (code && code >= 400 && code < 500) {
      type = CircuitBreakerErrorType.CLIENT;
      isRetryable = code === 408 || code === 429 || code >= 500;
    } else if (code && code >= 500) {
      type = CircuitBreakerErrorType.SERVER;
    }
    
    return {
      ...error,
      name: error.name || 'CircuitBreakerError',
      type,
      code,
      isRetryable,
      message: error.message || 'Unknown error occurred'
    };
  }
  
  private isCircuitBreakerError(error: Error): error is CircuitBreakerError {
    return 'type' in error && 'isRetryable' in error;
  }
  
  private createError(
    message: string,
    type: CircuitBreakerErrorType,
    code: string,
    isRetryable: boolean
  ): CircuitBreakerError {
    const error = new Error(message) as CircuitBreakerError;
    error.type = type;
    error.code = code;
    error.isRetryable = isRetryable;
    return error;
  }
}

// Export a default instance for convenience
export const circuitBreaker = new CircuitBreaker();

export default CircuitBreaker;
