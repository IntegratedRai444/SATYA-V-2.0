import { Router } from 'express';
import { successResponse } from '../../utils/apiResponse';
import { checkDatabaseConnection } from '../../services/database';
import { checkRedisConnection } from '../../services/cache';

const router = Router();

router.get('/', async (req, res) => {
  try {
    const [dbStatus, redisStatus] = await Promise.all([
      checkDatabaseConnection(),
      checkRedisConnection()
    ]);

    const status = dbStatus && redisStatus ? 'healthy' : 'degraded';
    
    return successResponse(res, {
      status,
      timestamp: new Date().toISOString(),
      services: {
        database: dbStatus ? 'connected' : 'disconnected',
        cache: redisStatus ? 'connected' : 'disconnected'
      },
      version: process.env.npm_package_version || '1.0.0'
    });
  } catch (error) {
    console.error('Health check failed:', error);
    return res.status(503).json({
      success: false,
      error: {
        code: 'SERVICE_UNAVAILABLE',
        message: 'Health check failed',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      services: {
        database: 'error',
        cache: 'error'
      },
      timestamp: new Date().toISOString()
    });
  }
});

export const healthRouter = router;
