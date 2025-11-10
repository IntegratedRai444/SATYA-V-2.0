import { Router } from 'express';
import { logger } from '../config';
import { pythonBridgeEnhanced as pythonBridge } from '../services/python-bridge-fixed';
import metrics from '../services/prometheus-metrics';
import alertingSystem from '../services/alerting-system';
import auditLogger from '../services/audit-logger';

const router = Router();

/**
 * Overall system health check
 */
router.get('/', async (req, res) => {
  try {
    const startTime = Date.now();
    
    // Check Python server health
    const pythonHealth = await pythonBridge.healthCheck();
    
    // Check database connectivity
    let dbHealth = { status: 'connected' };
    try {
      // Simple database check
      const { db } = await import('../db');
      await db.select().from(await import('@shared/schema').then(s => s.users)).limit(1);
    } catch (error) {
      dbHealth = { 
        status: 'error', 
        error: (error as Error).message 
      };
    }
    
    const responseTime = Date.now() - startTime;
    const overallHealthy = pythonHealth.status === 'connected' && dbHealth.status === 'connected';
    
    const healthStatus = {
      status: overallHealthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      components: {
        nodejs: {
          status: 'healthy',
          uptime: process.uptime(),
          memory: process.memoryUsage(),
          version: process.version
        },
        python: pythonHealth,
        database: dbHealth
      },
      metrics: {
        responseTime,
        uptime: process.uptime()
      }
    };
    
    const statusCode = overallHealthy ? 200 : 503;
    
    logger.info('Health check completed', {
      status: healthStatus.status,
      responseTime,
      pythonStatus: pythonHealth.status,
      dbStatus: dbHealth.status
    });
    
    res.status(statusCode).json(healthStatus);
    
  } catch (error) {
    logger.error('Health check failed', { error: (error as Error).message });
    
    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
      message: (error as Error).message
    });
  }
});

/**
 * Detailed system diagnostics
 */
router.get('/detailed', async (req, res) => {
  try {
    const startTime = Date.now();
    
    // Check all components
    const pythonHealth = await pythonBridge.healthCheck();
    
    // Check database
    let dbHealth = { status: 'connected', responseTime: 0 };
    try {
      const dbStartTime = Date.now();
      const { db } = await import('../db');
      await db.select().from(await import('@shared/schema').then(s => s.users)).limit(1);
      dbHealth.responseTime = Date.now() - dbStartTime;
    } catch (error) {
      dbHealth = { 
        status: 'error', 
        responseTime: Date.now() - startTime,
        error: (error as Error).message 
      } as any;
    }

    // Check WebSocket
    let wsHealth = { status: 'unknown', connections: 0 };
    try {
      const { webSocketManager } = await import('../services/websocket-manager');
      const stats = webSocketManager.getStats();
      wsHealth = {
        status: 'healthy',
        connections: stats.totalConnections,
        connectedUsers: stats.connectedUsers,
        averageConnectionsPerUser: stats.averageConnectionsPerUser
      } as any;
    } catch (error) {
      wsHealth = { status: 'error', error: (error as Error).message } as any;
    }

    // Check file system
    let fsHealth = { status: 'healthy', freeSpace: 0 };
    try {
      const fs = await import('fs');
      const stats = fs.statSync('.');
      fsHealth = {
        status: 'healthy',
        accessible: true
      } as any;
    } catch (error) {
      fsHealth = { status: 'error', error: (error as Error).message } as any;
    }

    const responseTime = Date.now() - startTime;
    const overallHealthy = pythonHealth.status === 'connected' && 
                          dbHealth.status === 'connected' &&
                          wsHealth.status === 'healthy';

    const healthStatus = {
      status: overallHealthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      responseTime,
      components: {
        nodejs: {
          status: 'healthy',
          uptime: process.uptime(),
          memory: {
            used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
            total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
            external: Math.round(process.memoryUsage().external / 1024 / 1024),
            rss: Math.round(process.memoryUsage().rss / 1024 / 1024)
          },
          cpu: process.cpuUsage(),
          version: process.version,
          platform: process.platform,
          arch: process.arch
        },
        python: pythonHealth,
        database: dbHealth,
        websocket: wsHealth,
        filesystem: fsHealth
      },
      metrics: {
        responseTime,
        uptime: process.uptime(),
        memoryUsage: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        cpuUsage: process.cpuUsage()
      }
    };
    
    const statusCode = overallHealthy ? 200 : 503;
    
    logger.info('Detailed health check completed', {
      status: healthStatus.status,
      responseTime,
      components: Object.keys(healthStatus.components).reduce((acc, key) => {
        acc[key] = (healthStatus.components as any)[key].status;
        return acc;
      }, {} as Record<string, string>)
    });
    
    res.status(statusCode).json(healthStatus);
    
  } catch (error) {
    logger.error('Detailed health check failed', { error: (error as Error).message });
    
    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: 'Detailed health check failed',
      message: (error as Error).message
    });
  }
});

/**
 * Prometheus metrics endpoint
 */
router.get('/metrics', (req, res) => {
  try {
    const metricsOutput = metrics.generatePrometheusFormat();
    res.set('Content-Type', 'text/plain; version=0.0.4; charset=utf-8');
    res.send(metricsOutput);
  } catch (error) {
    logger.error('Prometheus metrics generation failed', { error: (error as Error).message });
    res.status(500).json({
      error: 'Failed to generate metrics',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Performance metrics endpoint (JSON format)
 */
router.get('/performance', async (req, res) => {
  try {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    // Get WebSocket stats
    let wsStats = { totalConnections: 0, connectedUsers: 0 };
    try {
      const { webSocketManager } = await import('../services/websocket-manager');
      wsStats = webSocketManager.getStats();
    } catch (error) {
      // WebSocket stats not available
    }

    // Get session stats
    let sessionStats = { totalActiveSessions: 0 };
    try {
      const { sessionManager } = await import('../services/session-manager');
      sessionStats = sessionManager.getSessionStats();
    } catch (error) {
      // Session stats not available
    }

    const performanceMetrics = {
      timestamp: new Date().toISOString(),
      system: {
        uptime: process.uptime(),
        memory: {
          heapUsed: memUsage.heapUsed,
          heapTotal: memUsage.heapTotal,
          external: memUsage.external,
          rss: memUsage.rss,
          heapUsedMB: Math.round(memUsage.heapUsed / 1024 / 1024),
          heapTotalMB: Math.round(memUsage.heapTotal / 1024 / 1024),
          usagePercentage: Math.round((memUsage.heapUsed / memUsage.heapTotal) * 100)
        },
        cpu: {
          user: cpuUsage.user,
          system: cpuUsage.system
        }
      },
      application: {
        websocket: {
          totalConnections: wsStats.totalConnections,
          connectedUsers: wsStats.connectedUsers
        },
        sessions: {
          totalActiveSessions: sessionStats.totalActiveSessions
        }
      },
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        nodeEnv: process.env.NODE_ENV
      }
    };

    // Update metrics
    metrics.setGauge('memory_usage_bytes', memUsage.heapUsed, { type: 'heap_used' });
    metrics.setGauge('memory_usage_bytes', memUsage.heapTotal, { type: 'heap_total' });
    metrics.setGauge('memory_usage_bytes', memUsage.rss, { type: 'rss' });
    metrics.setGauge('active_websocket_connections', wsStats.totalConnections);
    metrics.setGauge('active_sessions', sessionStats.totalActiveSessions);

    res.json(performanceMetrics);
    
  } catch (error) {
    logger.error('Performance metrics collection failed', { error: (error as Error).message });
    
    res.status(500).json({
      error: 'Performance metrics collection failed',
      message: (error as Error).message
    });
  }
});

/**
 * Active alerts endpoint
 */
router.get('/alerts', (req, res) => {
  try {
    const alerts = alertingSystem.getActiveAlerts();
    res.json({
      count: alerts.length,
      alerts: alerts.map(alert => ({
        id: alert.id,
        severity: alert.severity,
        title: alert.title,
        message: alert.message,
        component: alert.component,
        timestamp: alert.timestamp
      }))
    });
  } catch (error) {
    logger.error('Failed to get alerts', { error: (error as Error).message });
    res.status(500).json({
      error: 'Failed to get alerts',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Alert history endpoint
 */
router.get('/alerts/history', (req, res) => {
  try {
    const hours = parseInt(req.query.hours as string) || 24;
    const alerts = alertingSystem.getAlertHistory(hours);
    
    res.json({
      count: alerts.length,
      hours,
      alerts: alerts.map(alert => ({
        id: alert.id,
        severity: alert.severity,
        title: alert.title,
        message: alert.message,
        component: alert.component,
        timestamp: alert.timestamp,
        resolved: alert.resolved,
        resolvedAt: alert.resolvedAt
      }))
    });
  } catch (error) {
    logger.error('Failed to get alert history', { error: (error as Error).message });
    res.status(500).json({
      error: 'Failed to get alert history',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Alerting system health
 */
router.get('/alerting', (req, res) => {
  try {
    const alertingHealth = alertingSystem.getSystemHealth();
    res.json(alertingHealth);
  } catch (error) {
    logger.error('Failed to get alerting system health', { error: (error as Error).message });
    res.status(500).json({
      error: 'Failed to get alerting system health',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Audit system health
 */
router.get('/audit', async (req, res) => {
  try {
    const auditHealth = await auditLogger.getHealth();
    res.json(auditHealth);
  } catch (error) {
    logger.error('Failed to get audit system health', { error: (error as Error).message });
    res.status(500).json({
      error: 'Failed to get audit system health',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Python server proxy health check
 */
router.get('/python', async (req, res) => {
  try {
    const pythonHealth = await pythonBridge.healthCheck();
    
    const statusCode = pythonHealth.status === 'connected' ? 200 : 503;
    res.status(statusCode).json(pythonHealth);
    
  } catch (error) {
    logger.error('Python health check failed', { error: (error as Error).message });
    
    res.status(503).json({
      status: 'error',
      error: 'Python health check failed',
      message: (error as Error).message
    });
  }
});

export default router;