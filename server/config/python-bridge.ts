import { pythonConfig } from './python-config';
import { logger } from './logger';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const axiosInstance = require('axios');

// Node.js globals for TypeScript
declare const setInterval: (callback: () => void, ms: number) => NodeJS.Timeout;
declare const NodeJS: {
  Timeout: unknown;
};

// 🔥 PRODUCTION: Readiness cache and auto-warmup system
interface ReadinessCache {
  ready: boolean;
  lastCheck: number;
  lastWarmup: number;
  warmupInProgress: boolean;
  retryCount: number;
}

class PythonBridgeManager {
  private readinessCache: ReadinessCache = {
    ready: false,
    lastCheck: 0,
    lastWarmup: 0,
    warmupInProgress: false,
    retryCount: 0
  };
  
  private readonly CACHE_DURATION = 15000; // 15 seconds cache
  private readonly MAX_WARMUP_RETRIES = 10;
  private readonly WARMUP_RETRY_DELAY = 2000; // 2 seconds
  
  constructor() {
    // Start background readiness polling
    this.startReadinessPolling();
    // Auto-warmup on startup
    this.initializeOnStartup();
  }
  
  /**
   * 🔥 PRODUCTION: Initialize Python service on Node startup
   */
  private async initializeOnStartup(): Promise<void> {
    logger.info('[PythonBridge] Starting Python service initialization...');
    
    try {
      // Check initial readiness
      const ready = await this.checkReadiness();
      if (!ready) {
        logger.info('[PythonBridge] Python not ready, triggering auto-warmup...');
        await this.autoWarmup();
      } else {
        logger.info('[PythonBridge] Python service ready on startup');
        this.readinessCache.ready = true;
      }
    } catch (error) {
      logger.error('[PythonBridge] Startup initialization failed:', error);
    }
  }
  
  /**
   * 🔥 PRODUCTION: Background readiness polling
   */
  private startReadinessPolling(): void {
    setInterval(async () => {
      try {
        await this.checkReadiness();
      } catch (error) {
        logger.warn('[PythonBridge] Readiness poll failed:', error);
      }
    }, 20000); // Poll every 20 seconds
  }
  
  /**
   * 🔥 PRODUCTION: Check Python readiness using /ready endpoint only
   */
  private async checkReadiness(): Promise<boolean> {
    const now = Date.now();
    
    // Return cached result if still valid
    if (now - this.readinessCache.lastCheck < this.CACHE_DURATION && this.readinessCache.ready) {
      return this.readinessCache.ready;
    }
    
    try {
      logger.info('[PythonBridge] Checking Python readiness...');
      
      const response = await axiosInstance.get(
        `${pythonConfig.apiUrl}/ready`,
        {
          timeout: 5000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      const isReady = response.status === 200 && response.data?.ready === true;
      
      this.readinessCache.ready = isReady;
      this.readinessCache.lastCheck = now;
      this.readinessCache.retryCount = 0;
      
      logger.info(`[PythonBridge] Ready: ${isReady}`, {
        responseTime: Date.now() - now,
        modelsLoaded: response.data?.models_loaded,
        memoryUsage: response.data?.memory_usage
      });
      
      return isReady;
      
    } catch (error) {
      this.readinessCache.ready = false;
      this.readinessCache.lastCheck = now;
      
      logger.warn('[PythonBridge] Readiness check failed:', {
        error: error instanceof Error ? error.message : 'Unknown error',
        url: `${pythonConfig.apiUrl}/ready`
      });
      
      return false;
    }
  }
  
  /**
   * 🔥 PRODUCTION: Auto-warmup Python service with retry logic
   */
  private async autoWarmup(): Promise<boolean> {
    if (this.readinessCache.warmupInProgress) {
      logger.info('[PythonBridge] Warmup already in progress');
      return false;
    }
    
    this.readinessCache.warmupInProgress = true;
    this.readinessCache.lastWarmup = Date.now();
    
    try {
      logger.info('[PythonBridge] Starting auto-warmup...');
      
      const response = await axiosInstance.get(
        `${pythonConfig.apiUrl}/warmup`,
        {
          timeout: 30000, // 30 seconds for warmup
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      logger.info('[PythonBridge] Warmup completed', {
        warmupTime: response.data?.results?.warmup_time,
        imageDetector: response.data?.results?.image_detector,
        modelsLoaded: response.data?.results?.models_loaded
      });
      
      // Poll readiness after warmup
      return await this.pollReadinessAfterWarmup();
      
    } catch (error) {
      logger.error('[PythonBridge] Warmup failed:', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      return false;
    } finally {
      this.readinessCache.warmupInProgress = false;
    }
  }
  
  /**
   * 🔥 PRODUCTION: Poll readiness after warmup with retries
   */
  private async pollReadinessAfterWarmup(): Promise<boolean> {
    for (let attempt = 1; attempt <= this.MAX_WARMUP_RETRIES; attempt++) {
      logger.info(`[PythonBridge] Readiness poll attempt ${attempt}/${this.MAX_WARMUP_RETRIES}`);
      
      await new Promise(resolve => setTimeout(resolve, this.WARMUP_RETRY_DELAY));
      
      const ready = await this.checkReadiness();
      if (ready) {
        logger.info(`[PythonBridge] Ready after warmup (attempt ${attempt})`);
        return true;
      }
      
      this.readinessCache.retryCount = attempt;
    }
    
    logger.error('[PythonBridge] Failed to achieve readiness after warmup');
    return false;
  }
  
  /**
   * 🔥 PRODUCTION: Public API - Get readiness with auto-warmup fallback
   */
  public async getReadiness(): Promise<boolean> {
    const ready = await this.checkReadiness();
    
    if (!ready && !this.readinessCache.warmupInProgress) {
      logger.info('[PythonBridge] Service not ready, triggering auto-warmup...');
      return await this.autoWarmup();
    }
    
    return ready;
  }
  
  /**
   * 🔥 PRODUCTION: Get cached readiness status
   */
  public getCachedReadiness(): boolean {
    return this.readinessCache.ready;
  }
  
  /**
   * 🔥 PRODUCTION: Force readiness refresh
   */
  public async refreshReadiness(): Promise<boolean> {
    this.readinessCache.lastCheck = 0; // Invalidate cache
    return await this.checkReadiness();
  }
}

// 🔥 PRODUCTION: Global bridge manager instance
export const bridgeManager = new PythonBridgeManager();

// 🔥 PRODUCTION: Health check interface
interface HealthCheckResult {
  healthy: boolean;
  responseTime: number;
  error?: string;
  timestamp: number;
  details?: Record<string, unknown>;
}

// 🔥 PRODUCTION: Python service health with readiness management
class PythonServiceHealth {
  private lastHealthCheck = 0;
  private healthCheckInterval = 30000; // 30 seconds
  private isHealthy = false;

  async checkHealth(): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    try {
      // 🔥 PRODUCTION: Use bridge manager for readiness
      const ready = await bridgeManager.getReadiness();
      const responseTime = Date.now() - startTime;
      
      this.isHealthy = ready;
      this.lastHealthCheck = Date.now();

      return {
        healthy: ready,
        responseTime,
        timestamp: this.lastHealthCheck,
        details: { ready, source: 'bridge_manager' }
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

// 🔥 PRODUCTION: Enhanced Python bridge with readiness management
export class EnhancedPythonBridge {
  private static _instance: EnhancedPythonBridge;
  
  static getInstance(): EnhancedPythonBridge {
    if (!EnhancedPythonBridge._instance) {
      EnhancedPythonBridge._instance = new EnhancedPythonBridge();
    }
    return EnhancedPythonBridge._instance;
  }
  
  /**
   * 🔥 PRODUCTION: Get health status using bridge manager
   */
  async getHealthStatus(): Promise<{ healthy: boolean; error?: string; details?: unknown }> {
    try {
      const ready = await bridgeManager.getReadiness();
      return {
        healthy: ready,
        details: { ready, source: 'bridge_manager' }
      };
    } catch (error) {
      return {
        healthy: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
  
  /**
   * 🔥 PRODUCTION: Check if Python is ready (cached)
   */
  isReady(): boolean {
    return bridgeManager.getCachedReadiness();
  }
  
  /**
   * 🔥 PRODUCTION: Force readiness refresh
   */
  async refreshReadiness(): Promise<boolean> {
    return await bridgeManager.refreshReadiness();
  }
}

// 🔥 PRODUCTION: Circuit breaker implementation
interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeout: number;
  monitoringPeriod: number;
  halfOpenMaxCalls: number;
}

interface CircuitBreakerStats {
  failures: number;
  successes: number;
  lastFailureTime: number;
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
}

class CircuitBreaker {
  private config: CircuitBreakerConfig;
  private stats: CircuitBreakerStats;
  private lastFailureTime = 0;
  
  constructor(config: CircuitBreakerConfig) {
    this.config = config;
    this.stats = {
      failures: 0,
      successes: 0,
      lastFailureTime: 0,
      state: 'CLOSED'
    };
  }
  
  recordFailure(): void {
    this.stats.failures++;
    this.stats.lastFailureTime = Date.now();
    if (this.stats.failures >= this.config.failureThreshold) {
      this.stats.state = 'OPEN';
    }
  }
  
  recordSuccess(): void {
    this.stats.successes++;
    this.stats.failures = 0;
    this.stats.state = 'CLOSED';
  }
  
  recordAttempt(): void {
    // Track attempts for monitoring
  }
  
  isOpen(): boolean {
    return this.stats.state === 'OPEN';
  }
  
  getStats(): CircuitBreakerStats {
    return { ...this.stats };
  }
  
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.isOpen()) {
      throw new Error('Circuit breaker is OPEN');
    }
    
    try {
      const result = await fn();
      this.recordSuccess();
      return result;
    } catch (error) {
      this.recordFailure();
      throw error;
    }
  }
}

// 🔥 PRODUCTION: Export instances
export const circuitBreaker = new CircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 60000,  // 1 minute
  monitoringPeriod: 120000, // 2 minutes
  halfOpenMaxCalls: 2
});

export const healthChecker = new PythonServiceHealth();

// 🔥 PRODUCTION: Export the enhanced bridge instance
export const enhancedPythonBridge = EnhancedPythonBridge.getInstance();
