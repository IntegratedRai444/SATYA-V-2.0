import { Router, Request, Response } from 'express';
import { logger } from '../config/logger';
import { enhancedPythonBridge } from '../config/python-bridge';
import { circuitBreaker } from '../config/python-bridge';
import { supabaseAdmin } from '../config/supabase';

const router = Router();

// System health check endpoint
router.get('/health', async (req: Request, res: Response) => {
  try {
    const healthStatus = {
      status: 'healthy' as 'healthy' | 'degraded' | 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: '2.0.0',
      checks: {
        database: {
          status: 'unknown' as 'healthy' | 'unhealthy',
          response_time: 0,
          error: ''
        },
        python_service: {
          status: 'unknown' as 'healthy' | 'unhealthy',
          response_time: 0,
          error: '',
          details: {}
        },
        circuit_breaker: {
          status: 'unknown' as 'healthy' | 'degraded',
          state: '',
          failure_count: 0,
          last_failure_time: 0
        },
        memory: {
          status: 'unknown' as 'healthy',
          heap_used: '',
          heap_total: '',
          external: ''
        }
      }
    };

    // Check database connectivity
    try {
      const startTime = Date.now();
      const { error } = await supabaseAdmin
        .from('scans')
        .select('id')
        .is('deleted_at', null)  
        .limit(1)
        .single();
      
      if (error) {
        healthStatus.checks.database = {
          status: 'unhealthy',
          response_time: Date.now() - startTime,
          error: error.message
        };
        healthStatus.status = 'degraded';
      } else {
        healthStatus.checks.database = {
          status: 'healthy',
          response_time: Date.now() - startTime,
          error: ''
        };
      }
    } catch (dbError) {
      healthStatus.checks.database = {
        status: 'unhealthy',
        response_time: 0,
        error: (dbError as Error).message
      };
      healthStatus.status = 'degraded';
    }

    // Check Python service health
    try {
      const startTime = Date.now();
      const pythonHealth = await enhancedPythonBridge.request({
        method: 'GET',
        url: '/health',
        timeout: 5000
      });
      
      healthStatus.checks.python_service = {
        status: 'healthy',
        response_time: Date.now() - startTime,
        error: '',
        details: pythonHealth || {}
      };
    } catch (pythonError) {
      healthStatus.checks.python_service = {
        status: 'unhealthy',
        response_time: 0,
        error: (pythonError as Error).message,
        details: {}
      };
      healthStatus.status = 'degraded';
    }

    // Check circuit breaker state
    const breakerState = circuitBreaker.getState();
    const stats = circuitBreaker.getStats();
    healthStatus.checks.circuit_breaker = {
      status: breakerState === 'CLOSED' ? 'healthy' : 'degraded',
      state: breakerState,
      failure_count: stats.failures,
      last_failure_time: stats.lastFailureTime
    };

    // Check memory usage
    const memUsage = process.memoryUsage();
    healthStatus.checks.memory = {
      status: 'healthy',
      heap_used: Math.round(memUsage.heapUsed / 1024 / 1024 * 100) + '%',
      heap_total: Math.round(memUsage.heapTotal / 1024 / 1024 * 100) + '%',
      external: Math.round(memUsage.external / 1024 / 1024 * 100) + '%'
    };

    // Overall status determination
    const allHealthy = Object.values(healthStatus.checks)
      .every(check => check.status === 'healthy');

    if (!allHealthy) {
      healthStatus.status = 'degraded';
    }

    // Log health status
    logger.info('System health check', {
      status: healthStatus.status,
      checks: healthStatus.checks
    });

    // Return appropriate HTTP status
    const statusCode = allHealthy ? 200 : 503;
    
    res.status(statusCode).json(healthStatus);
    
  } catch (error) {
    logger.error('System health check failed', { error: (error as Error).message });
    
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: (error as Error).message,
      checks: {}
    });
  }
});

// System info endpoint
router.get('/info', async (req: Request, res: Response) => {
  try {
    const infoData = {
      service: 'satyaai-node',
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      platform: process.platform,
      nodeVersion: process.version,
      timestamp: new Date().toISOString(),
      endpoints: {
        health: '/system/health',
        info: '/system/info',
        analysis: '/api/v2/analysis',
        dashboard: '/api/v2/dashboard'
      }
    };
    
    res.json(infoData);
    
  } catch (error) {
    logger.error('System info check failed:', error);
    res.status(500).json({
      error: 'System info failed',
      timestamp: new Date().toISOString()
    });
  }
});

export { router as systemRouter };
