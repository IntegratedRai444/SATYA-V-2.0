import { Request, Response } from 'express';
import { logger } from '../config/logger';
import { checkDatabaseConnection } from '../services/database';
import { checkPythonService } from '../services/python-http-bridge';

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
    // Add more checks as needed
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

    const isHealthy = Object.values(checks).every(Boolean);
    const uptime = process.uptime();
    const responseTime = Date.now() - startTime;

    const healthData = {
      status: isHealthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      uptime: Math.floor(uptime * 1000), // Convert to milliseconds
      version: process.env.npm_package_version || 'unknown',
      checks,
      responseTime: `${responseTime}ms`
    };

    // Log health check
    logger.info('Health check', {
      ...healthData,
      ip: req.ip,
      userAgent: req.get('user-agent')
    });

    // Return appropriate status code based on health status
    return res.status(isHealthy ? 200 : 503).json(healthData);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Health check failed', { 
      error: errorMessage,
      stack: error instanceof Error ? error.stack : undefined
    });

    return res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Service Unavailable',
      details: process.env.NODE_ENV === 'production' ? undefined : errorMessage
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
