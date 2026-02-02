import { logger } from '../config/logger';

export interface CircuitBreakerConfig {
  failureThreshold: number;      // Number of failures before opening
  recoveryTimeout: number;       // Time to wait before trying again (ms)
  monitoringPeriod: number;      // Time window for failure counting (ms)
  expectedRecoveryTime: number;  // Expected time for recovery (ms)
}

export interface CircuitBreakerState {
  failures: number;
  lastFailureTime: number;
  nextAttempt: number;
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
}

export class CircuitBreaker {
  private state: CircuitBreakerState;
  private config: CircuitBreakerConfig;
  private name: string;

  constructor(name: string, config: Partial<CircuitBreakerConfig> = {}) {
    this.name = name;
    this.config = {
      failureThreshold: 5,
      recoveryTimeout: 60000,        // 1 minute
      monitoringPeriod: 10000,      // 10 seconds
      expectedRecoveryTime: 30000,  // 30 seconds
      ...config
    };

    this.state = {
      failures: 0,
      lastFailureTime: 0,
      nextAttempt: 0,
      state: 'CLOSED'
    };

    logger.info(`CircuitBreaker '${name}' initialized`, this.config);
  }

  // Execute an operation with circuit breaker protection
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    const now = Date.now();

    // Check if circuit is open and recovery time has passed
    if (this.state.state === 'OPEN' && now >= this.state.nextAttempt) {
      logger.info(`CircuitBreaker '${this.name}' transitioning to HALF_OPEN`);
      this.state.state = 'HALF_OPEN';
      this.state.failures = 0;
    }

    // Reject calls if circuit is open
    if (this.state.state === 'OPEN') {
      const error = new Error(`CircuitBreaker '${this.name}' is OPEN`) as Error & { code: string; retryAfter: number };
      error.code = 'CIRCUIT_BREAKER_OPEN';
      error.retryAfter = this.state.nextAttempt - now;
      throw error;
    }

    // Reset failure count if monitoring period has passed
    if (now - this.state.lastFailureTime > this.config.monitoringPeriod) {
      this.state.failures = 0;
    }

    try {
      // Execute the operation
      const result = await operation();

      // Success: reset failures and close circuit if half-open
      if (this.state.state === 'HALF_OPEN') {
        logger.info(`CircuitBreaker '${this.name}' closing after successful operation`);
        this.state.state = 'CLOSED';
        this.state.failures = 0;
      }

      return result;
    } catch (error) {
      // Failure: increment failures and potentially open circuit
      this.state.failures++;
      this.state.lastFailureTime = now;

      logger.warn(`CircuitBreaker '${this.name}' operation failed`, {
        failures: this.state.failures,
        threshold: this.config.failureThreshold,
        error: error instanceof Error ? error.message : error
      });

      if (this.state.failures >= this.config.failureThreshold) {
        this.openCircuit();
      }

      throw error;
    }
  }

  // Manually open the circuit
  openCircuit(): void {
    if (this.state.state !== 'OPEN') {
      logger.warn(`CircuitBreaker '${this.name}' opening due to failure threshold`);
      this.state.state = 'OPEN';
      this.state.nextAttempt = Date.now() + this.config.recoveryTimeout;
    }
  }

  // Manually close the circuit
  closeCircuit(): void {
    logger.info(`CircuitBreaker '${this.name}' manually closed`);
    this.state.state = 'CLOSED';
    this.state.failures = 0;
    this.state.lastFailureTime = 0;
    this.state.nextAttempt = 0;
  }

  // Get current state
  getState(): CircuitBreakerState {
    return { ...this.state };
  }

  // Get configuration
  getConfig(): CircuitBreakerConfig {
    return { ...this.config };
  }

  // Check if circuit is closed (allowing requests)
  isClosed(): boolean {
    return this.state.state === 'CLOSED';
  }

  // Check if circuit is open (blocking requests)
  isOpen(): boolean {
    return this.state.state === 'OPEN';
  }

  // Get time until next attempt (when circuit is open)
  getTimeUntilNextAttempt(): number {
    if (this.state.state !== 'OPEN') {
      return 0;
    }
    return Math.max(0, this.state.nextAttempt - Date.now());
  }

  // Get health status for monitoring
  getHealthStatus(): {
    name: string;
    state: string;
    failures: number;
    isHealthy: boolean;
    timeUntilNextAttempt?: number;
  } {
    return {
      name: this.name,
      state: this.state.state,
      failures: this.state.failures,
      isHealthy: this.state.state === 'CLOSED',
      timeUntilNextAttempt: this.getTimeUntilNextAttempt() || undefined
    };
  }
}

// Circuit breaker registry for managing multiple breakers
class CircuitBreakerRegistry {
  private breakers = new Map<string, CircuitBreaker>();

  // Get or create a circuit breaker
  get(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      this.breakers.set(name, new CircuitBreaker(name, config));
    }
    return this.breakers.get(name)!;
  }

  // Get all circuit breakers
  getAll(): CircuitBreaker[] {
    return Array.from(this.breakers.values());
  }

  // Get health status for all breakers
  getAllHealthStatus(): Array<ReturnType<CircuitBreaker['getHealthStatus']>> {
    return this.getAll().map(breaker => breaker.getHealthStatus());
  }

  // Close all circuit breakers (useful for recovery)
  closeAll(): void {
    logger.info('Closing all circuit breakers');
    this.getAll().forEach(breaker => breaker.closeCircuit());
  }

  // Reset all circuit breakers
  reset(): void {
    logger.info('Resetting all circuit breakers');
    this.breakers.clear();
  }
}

// Export singleton registry
export const circuitBreakerRegistry = new CircuitBreakerRegistry();

// Export default circuit breaker for Python service
export const pythonServiceCircuitBreaker = circuitBreakerRegistry.get('python-service', {
  failureThreshold: 3,
  recoveryTimeout: 30000,        // 30 seconds
  monitoringPeriod: 60000,      // 1 minute
  expectedRecoveryTime: 15000,  // 15 seconds
});
