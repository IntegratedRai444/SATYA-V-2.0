import { pythonHealthChecker } from './health-checker';
import { logger } from './logger';

interface StrictModeValidation {
  valid: boolean;
  error?: string;
  details?: {
    models_loaded: boolean;
    gpu_available: boolean;
    service_healthy: boolean;
  };
}

class StrictModeGuard {
  private static instance: StrictModeGuard;
  private lastValidation: number = 0;
  private lastResult: StrictModeValidation | null = null;
  private validationInterval: number = 30000; // 30 seconds

  static getInstance(): StrictModeGuard {
    if (!StrictModeGuard.instance) {
      StrictModeGuard.instance = new StrictModeGuard();
    }
    return StrictModeGuard.instance;
  }

  async validateStrictMode(): Promise<StrictModeValidation> {
    const now = Date.now();
    
    // Cache recent validations
    if (this.lastResult && (now - this.lastValidation) < this.validationInterval) {
      return this.lastResult;
    }

    const isStrictMode = process.env.STRICT_DEEPFAKE_MODE === 'true';
    
    if (!isStrictMode) {
      const result: StrictModeValidation = {
        valid: true,
        details: {
          models_loaded: true,
          gpu_available: true,
          service_healthy: true
        }
      };
      
      this.lastValidation = now;
      this.lastResult = result;
      
      return result;
    }

    // Strict mode validation
    try {
      const health = await pythonHealthChecker.checkHealth();
      
      const validation: StrictModeValidation = {
        valid: health.healthy && Boolean(health.details?.models_loaded),
        details: {
          models_loaded: Boolean(health.details?.models_loaded),
          gpu_available: Boolean(health.details?.gpu_available),
          service_healthy: health.healthy
        }
      };

      if (!validation.valid) {
        if (!health.healthy) {
          validation.error = 'ML service is unhealthy';
        } else if (!health.details?.models_loaded) {
          validation.error = 'ML models are not loaded';
        }
      }

      this.lastValidation = now;
      this.lastResult = validation;

      return validation;

    } catch (error) {
      const result: StrictModeValidation = {
        valid: false,
        error: error instanceof Error ? error.message : 'Unknown validation error',
        details: {
          models_loaded: false,
          gpu_available: false,
          service_healthy: false
        }
      };

      this.lastValidation = now;
      this.lastResult = result;

      return result;
    }
  }

  async checkBeforeRequest(): Promise<boolean> {
    const validation = await this.validateStrictMode();
    
    if (!validation.valid) {
      logger.error('🚫 Strict mode validation failed', {
        error: validation.error,
        details: validation.details
      });
      
      throw new Error(`Strict mode validation failed: ${validation.error}`);
    }

    return true;
  }

  getValidationStatus(): StrictModeValidation | null {
    return this.lastResult;
  }

  forceValidation(): Promise<StrictModeValidation> {
    this.lastValidation = 0; // Reset cache
    return this.validateStrictMode();
  }
}

export const strictModeGuard = StrictModeGuard.getInstance();
