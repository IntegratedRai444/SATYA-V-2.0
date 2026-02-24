import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import { enhancedPythonBridge } from '../config/python-bridge';
import { captureError } from '../services/error-monitor';
import { setInterval } from 'timers';

interface ServiceHealth {
  python: boolean;
  database: boolean;
  websocket: boolean;
}

interface FallbackConfig {
  enablePythonFallback: boolean;
  enableDatabaseFallback: boolean;
  enableWebSocketFallback: boolean;
  pythonFailureMessage: string;
  databaseFailureMessage: string;
  webSocketFailureMessage: string;
}

class FallbackManager {
  private serviceHealth: ServiceHealth = {
    python: false,
    database: false,
    websocket: false
  };

  private config: FallbackConfig = {
    enablePythonFallback: true,
    enableDatabaseFallback: true,
    enableWebSocketFallback: true,
    pythonFailureMessage: 'AI analysis service temporarily unavailable. Please try again later.',
    databaseFailureMessage: 'Database temporarily unavailable. Your analysis is still being processed.',
    webSocketFailureMessage: 'Real-time updates temporarily unavailable. Results will be available upon completion.'
  };

  // Check service health
  async checkServiceHealth(): Promise<void> {
    try {
      // Check Python service health
      const pythonHealth = await enhancedPythonBridge.getHealthStatus();
      this.serviceHealth.python = pythonHealth.healthy;

      // Check database connectivity (simple check)
      // Note: In production, you'd check actual database connection
      // For now, we'll assume it's available if Python is available
      this.serviceHealth.database = this.serviceHealth.python;

      // WebSocket health is handled by the websocket manager
      this.serviceHealth.websocket = true; // Assume healthy if server is running

      logger.info('[FALLBACK] Service health check completed', {
        python: this.serviceHealth.python,
        database: this.serviceHealth.database,
        websocket: this.serviceHealth.websocket
      });

    } catch (error) {
      logger.error('[FALLBACK] Health check failed', error);
      // Assume services are unhealthy if health check fails
      this.serviceHealth.python = false;
      this.serviceHealth.database = false;
      this.serviceHealth.websocket = false;
    }
  }

  // Middleware function
  createMiddleware(config: Partial<FallbackConfig> = {}) {
    const finalConfig = { ...this.config, ...config };

    return (req: Request, res: Response, next: NextFunction) => {
      // Skip fallback for health check endpoints
      if (req.path.includes('/health') || req.path.includes('/metrics')) {
        return next();
      }

      // Check if this is a critical analysis request
      if (req.path.includes('/analysis') && req.method === 'POST') {
        this.checkServiceHealth();

        // Python service fallback
        if (finalConfig.enablePythonFallback && !this.serviceHealth.python) {
          logger.warn('[FALLBACK] Python service down, returning fallback response', {
            path: req.path,
            method: req.method
          });

          return res.status(503).json({
            success: false,
            error: {
              code: 'AI_SERVICE_UNAVAILABLE',
              message: finalConfig.pythonFailureMessage,
              retryAfter: 60
            },
            fallback: {
                type: 'python_service_down',
                message: 'AI analysis service is temporarily unavailable'
              }
          });
        }

        // Database fallback
        if (finalConfig.enableDatabaseFallback && !this.serviceHealth.database) {
          logger.warn('[FALLBACK] Database service down, returning fallback response', {
            path: req.path,
            method: req.method
          });

          return res.status(503).json({
            success: false,
            error: {
              code: 'DATABASE_UNAVAILABLE',
              message: finalConfig.databaseFailureMessage,
              retryAfter: 30
            },
            fallback: {
                type: 'database_down',
                message: 'Database service is temporarily unavailable'
              }
          });
        }

        // WebSocket fallback (for WebSocket upgrade requests)
        if (finalConfig.enableWebSocketFallback && !this.serviceHealth.websocket && req.headers.upgrade === 'websocket') {
          logger.warn('[FALLBACK] WebSocket service down, rejecting connection', {
            path: req.path,
            method: req.method
          });

          return res.status(503).json({
            success: false,
            error: {
              code: 'WEBSOCKET_UNAVAILABLE',
              message: finalConfig.webSocketFailureMessage,
              retryAfter: 30
            },
            fallback: {
              type: 'websocket_down',
              message: 'Real-time updates temporarily unavailable'
            }
          });
        }
      }

      // For non-critical requests, proceed normally
      next();
    };
  }

  // Get current service health
  getServiceHealth(): ServiceHealth {
    return { ...this.serviceHealth };
  }

  // Update configuration
  updateConfig(newConfig: Partial<FallbackConfig>): void {
    this.config = { ...this.config, ...newConfig };
    logger.info('[FALLBACK] Configuration updated', this.config);
  }
}

// Singleton instance
const fallbackManager = new FallbackManager();
export default fallbackManager;

// Export middleware function
export const createFallbackMiddleware = (config?: Partial<FallbackConfig>) => 
  fallbackManager.createMiddleware(config);

// Export health checker
export const getServiceHealth = () => fallbackManager.getServiceHealth();

// Export config updater
export const updateFallbackConfig = (config: Partial<FallbackConfig>) => 
  fallbackManager.updateConfig(config);
