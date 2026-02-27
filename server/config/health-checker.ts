import axios from 'axios';
import { pythonConfig } from './python-config';
import { logger } from './logger';

// Create a typed axios instance
const httpClient = axios as any;

interface HealthCheckResult {
  healthy: boolean;
  responseTime: number;
  error?: string;
  details?: {
    models_loaded: boolean;
    gpu_available: boolean;
    service_uptime: number;
  };
}

class PythonHealthChecker {
  private lastCheck: number = 0;
  private lastResult: HealthCheckResult | null = null;
  private checkInterval: number = 30000; // 30 seconds
  private timeout: number = 10000; // 10 seconds

  async checkHealth(): Promise<HealthCheckResult> {
    const now = Date.now();
    
    // Cache recent results to avoid hammering the service
    if (this.lastResult && (now - this.lastCheck) < this.checkInterval) {
      return this.lastResult;
    }

    const startTime = Date.now();
    
    try {
      const response = await httpClient.get(`${pythonConfig.apiUrl}/health`, {
        timeout: this.timeout,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'SatyaAI-Node-Bridge/1.0'
        }
      });

      const responseTime = Date.now() - startTime;
      const healthy = response.status === 200 && response.data?.status === 'healthy';

      const result: HealthCheckResult = {
        healthy,
        responseTime,
        details: response.data || {}
      };

      this.lastCheck = now;
      this.lastResult = result;

      if (healthy) {
        logger.info('✅ Python service health check passed', {
          responseTime,
          details: result.details
        });
      } else {
        logger.warn('⚠️ Python service health check failed', {
          responseTime,
          status: response.data?.status,
          details: result.details
        });
      }

      return result;

    } catch (error) {
      const responseTime = Date.now() - startTime;
      const result: HealthCheckResult = {
        healthy: false,
        responseTime,
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      this.lastCheck = now;
      this.lastResult = result;

      logger.error('❌ Python service health check failed', {
        responseTime,
        error: result.error
      });

      return result;
    }
  }

  async validateBeforeRequest(): Promise<boolean> {
    const health = await this.checkHealth();
    
    if (!health.healthy) {
      logger.error('🚫 Python service unhealthy - blocking request', {
        error: health.error,
        responseTime: health.responseTime
      });
      return false;
    }

    // Additional validation for strict mode
    if (process.env.STRICT_DEEPFAKE_MODE === 'true') {
      if (!health.details?.models_loaded) {
        logger.error('🚫 Strict mode: ML models not loaded - blocking request');
        return false;
      }
    }

    return true;
  }

  getHealthStatus(): HealthCheckResult | null {
    return this.lastResult;
  }

  forceCheck(): Promise<HealthCheckResult> {
    this.lastCheck = 0; // Reset cache
    return this.checkHealth();
  }
}

// Export singleton instance
export const pythonHealthChecker = new PythonHealthChecker();
