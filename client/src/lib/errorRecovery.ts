/**
 * Error Recovery Mechanisms
 * Provides automatic retry logic and recovery strategies for failed operations
 */

import logger from './logger';
import { classifyError, type ErrorType } from './errorHandler';

// ============================================================================
// Types
// ============================================================================

export interface RetryOptions {
  maxAttempts?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  retryableErrors?: ErrorType[];
  onRetry?: (attempt: number, error: Error) => void;
  shouldRetry?: (error: Error, attempt: number) => boolean;
}

export interface RetryState {
  attempt: number;
  lastError: Error | null;
  nextRetryAt: number | null;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_RETRY_OPTIONS: Required<Omit<RetryOptions, 'onRetry' | 'shouldRetry'>> = {
  maxAttempts: 3,
  initialDelay: 1000, // 1 second
  maxDelay: 30000, // 30 seconds
  backoffMultiplier: 2,
  retryableErrors: ['network', 'timeout', 'server', 'rate_limit'],
};

// ============================================================================
// Retry with Exponential Backoff
// ============================================================================

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const config = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      // Check if we should retry
      const shouldRetry = options.shouldRetry
        ? options.shouldRetry(lastError, attempt)
        : shouldRetryError(lastError, config.retryableErrors);

      if (!shouldRetry || attempt >= config.maxAttempts) {
        throw lastError;
      }

      // Calculate delay with exponential backoff
      const delay = Math.min(
        config.initialDelay * Math.pow(config.backoffMultiplier, attempt - 1),
        config.maxDelay
      );

      logger.info(`Retrying operation (attempt ${attempt}/${config.maxAttempts})`, {
        delay,
        error: lastError.message,
      });

      // Call onRetry callback
      if (options.onRetry) {
        options.onRetry(attempt, lastError);
      }

      // Wait before retrying
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Check if error should be retried
 */
function shouldRetryError(error: Error, retryableErrors: ErrorType[]): boolean {
  const classified = classifyError(error);
  return classified.retryable && retryableErrors.includes(classified.type);
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Retry Manager
// ============================================================================

/**
 * Manages retry state for multiple operations
 */
export class RetryManager {
  private retryStates = new Map<string, RetryState>();

  /**
   * Execute operation with retry
   */
  async execute<T>(
    key: string,
    fn: () => Promise<T>,
    options?: RetryOptions
  ): Promise<T> {
    try {
      const result = await retryWithBackoff(fn, {
        ...options,
        onRetry: (attempt, error) => {
          this.updateState(key, {
            attempt,
            lastError: error,
            nextRetryAt: Date.now() + (options?.initialDelay || 1000),
          });
          options?.onRetry?.(attempt, error);
        },
      });

      // Clear state on success
      this.clearState(key);
      return result;
    } catch (error) {
      // Update state with final error
      this.updateState(key, {
        attempt: options?.maxAttempts || 3,
        lastError: error as Error,
        nextRetryAt: null,
      });
      throw error;
    }
  }

  /**
   * Get retry state for operation
   */
  getState(key: string): RetryState {
    return this.retryStates.get(key) || {
      attempt: 0,
      lastError: null,
      nextRetryAt: null,
    };
  }

  /**
   * Update retry state
   */
  private updateState(key: string, state: RetryState): void {
    this.retryStates.set(key, state);
  }

  /**
   * Clear retry state
   */
  clearState(key: string): void {
    this.retryStates.delete(key);
  }

  /**
   * Clear all retry states
   */
  clearAll(): void {
    this.retryStates.clear();
  }

  /**
   * Get all retry states
   */
  getAllStates(): Map<string, RetryState> {
    return new Map(this.retryStates);
  }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

export interface CircuitBreakerOptions {
  failureThreshold?: number;
  resetTimeout?: number;
  monitoringPeriod?: number;
}

export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Circuit breaker pattern implementation
 */
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failureCount = 0;
  private lastFailureTime: number | null = null;
  private successCount = 0;
  
  private readonly failureThreshold: number;
  private readonly resetTimeout: number;

  constructor(options: CircuitBreakerOptions = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 1 minute
  }

  /**
   * Execute function with circuit breaker
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (this.shouldAttemptReset()) {
        this.state = 'half-open';
        logger.info('Circuit breaker entering half-open state');
      } else {
        throw new Error('Circuit breaker is open');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Handle successful execution
   */
  private onSuccess(): void {
    this.failureCount = 0;
    
    if (this.state === 'half-open') {
      this.successCount++;
      if (this.successCount >= 2) {
        this.state = 'closed';
        this.successCount = 0;
        logger.info('Circuit breaker closed');
      }
    }
  }

  /**
   * Handle failed execution
   */
  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.successCount = 0;

    if (this.failureCount >= this.failureThreshold) {
      this.state = 'open';
      logger.warn('Circuit breaker opened', {
        failureCount: this.failureCount,
        threshold: this.failureThreshold,
      });
    }
  }

  /**
   * Check if should attempt reset
   */
  private shouldAttemptReset(): boolean {
    if (!this.lastFailureTime) return false;
    return Date.now() - this.lastFailureTime >= this.resetTimeout;
  }

  /**
   * Get current state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Reset circuit breaker
   */
  reset(): void {
    this.state = 'closed';
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = null;
  }
}

// ============================================================================
// Singleton Instances
// ============================================================================

export const retryManager = new RetryManager();

// ============================================================================
// Exported Utilities
// ============================================================================

/**
 * Create a retry wrapper for a function
 */
export function withRetry<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: RetryOptions
): T {
  return (async (...args: Parameters<T>) => {
    return retryWithBackoff(() => fn(...args), options);
  }) as T;
}

/**
 * Create a circuit breaker wrapper for a function
 */
export function withCircuitBreaker<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: CircuitBreakerOptions
): T {
  const breaker = new CircuitBreaker(options);
  
  return (async (...args: Parameters<T>) => {
    return breaker.execute(() => fn(...args));
  }) as T;
}

export default {
  retryWithBackoff,
  retryManager,
  RetryManager,
  CircuitBreaker,
  withRetry,
  withCircuitBreaker,
};
