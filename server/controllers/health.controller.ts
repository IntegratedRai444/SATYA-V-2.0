import { Request, Response } from 'express';
import { logger } from '../config/logger';
import { checkDatabaseConnection } from '../services/database';
import { checkPythonService } from '../services/python-http-bridge';
import { circuitBreakerRegistry } from '../services/circuit-breaker';
import { errorMetrics } from '../services/error-metrics';
import webSocketManager from '../services/websocket-manager';

/**
 * Health check controller
 * @param req Express request
 * @param res Express response
 */
export const healthCheck = async (req: Request, res: Response) => {
  const startTime = Date.now();
  const checks = {
    database: false,
    pythonService: false,
    webSocket: false,
    circuitBreakers: false,
    memory: false,
    disk: false,
    errorRate: false
  };

  try {
    // Check database connection
    checks.database = await checkDatabaseConnection();
    
    // Check Python service if needed
    if (process.env.ENABLE_PYTHON_SERVICE === 'true') {
      checks.pythonService = await checkPythonService();
    } else {
      checks.pythonService = true; // Skip check if not enabled
    }

    // Check WebSocket service
    checks.webSocket = webSocketManager.isHealthy();

    // Check circuit breakers
    const circuitBreakerHealth = circuitBreakerRegistry.getAllHealthStatus();
    checks.circuitBreakers = circuitBreakerHealth.every(cb => cb.isHealthy);

    // Check memory usage
    const memUsage = process.memoryUsage();
    const memoryThreshold = 1024 * 1024 * 1024; // 1GB
    checks.memory = memUsage.heapUsed < memoryThreshold;

    // Check disk space (basic check)
    const diskUsage = process.resourceUsage();
    checks.disk = diskUsage.userCPUTime < 10000000000; // 10 seconds CPU time

    // Check error rate
    const errorRate = errorMetrics.getErrorRate(5); // Last 5 minutes
    checks.errorRate = errorRate < 10; // Less than 10 errors per minute

    const isHealthy = Object.values(checks).every(Boolean);
    const responseTime = Date.now() - startTime;

    // Detailed health information
    const healthInfo = {
      status: isHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      responseTime: `${responseTime}ms`,
      uptime: `${Math.floor(process.uptime())}s`,
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      checks,
      details: {
        memory: {
          used: `${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`,
          total: `${Math.round(memUsage.heapTotal / 1024 / 1024)}MB`,
          rss: `${Math.round(memUsage.rss / 1024 / 1024)}MB`
        },
        circuitBreakers: circuitBreakerHealth.map(cb => ({
          name: cb.name,
          state: cb.state,
          failures: cb.failures,
          isHealthy: cb.isHealthy
        })),
        errorMetrics: {
          errorRate: `${errorRate.toFixed(2)}/min`,
          totalErrors: errorMetrics.getMetricsSummary().totalErrors,
          topErrors: errorMetrics.getMetricsSummary().topErrors.slice(0, 5)
        },
        webSocket: {
          connectedClients: webSocketManager.getConnectedClientsCount(),
          isHealthy: webSocketManager.isHealthy()
        }
      }
    };

    if (isHealthy) {
      res.status(200).json(healthInfo);
    } else {
      res.status(503).json({
        ...healthInfo,
        status: 'unhealthy',
        issues: Object.entries(checks)
          .filter(([, healthy]) => !healthy)
          .map(([name]) => name)
      });
    }
  } catch (error) {
    logger.error('Health check failed', error);
    res.status(500).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
};

/**
 * Readiness check endpoint
 */
export const readinessCheck = async (req: Request, res: Response) => {
  try {
    // Check if all critical services are ready
    const checks = await Promise.allSettled([
      checkDatabaseConnection(),
      checkPythonService(),
    ]);

    const allReady = checks.every(check => 
      check.status === 'fulfilled' && check.value === true
    );

    if (allReady) {
      res.status(200).json({
        status: 'ready',
        timestamp: new Date().toISOString(),
      });
    } else {
      res.status(503).json({
        status: 'not ready',
        timestamp: new Date().toISOString(),
        checks: checks.map(check => ({
          status: check.status,
          value: check.status === 'fulfilled' ? check.value : 'failed'
        }))
      });
    }
  } catch (error) {
    logger.error('Readiness check failed:', error);
    res.status(503).json({
      status: 'not ready',
      timestamp: new Date().toISOString(),
      error: 'Readiness check failed',
    });
  }
};

/**
 * Liveness check endpoint
 */
export const livenessCheck = async (req: Request, res: Response) => {
  try {
    // Simple liveness check - just respond quickly
    res.status(200).json({
      status: 'alive',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    });
  } catch (error) {
    logger.error('Liveness check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Liveness check failed',
    });
  }
};

// Export for testing
export const healthController = { 
  healthCheck,
  readinessCheck,
  livenessCheck 
};
