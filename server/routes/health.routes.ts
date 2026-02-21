import { Router } from 'express';
import { healthController } from '../controllers/health.controller';
import { checkDatabaseConnection } from '../services/database';

export const healthRouter = Router();

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Health check endpoint
 *     description: Returns the health status of the application
 *     responses:
 *       200:
 *         description: Application is healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: 'healthy'
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                   example: '2023-01-01T00:00:00.000Z'
 *                 uptime:
 *                   type: number
 *                   description: Uptime in seconds
 *                   example: 123.45
 *                 version:
 *                   type: string
 *                   example: '1.0.0'
 *       503:
 *         description: Service Unavailable
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
healthRouter.get('/', healthController.healthCheck);

/**
 * @swagger
 * /health/ready:
 *   get:
 *     summary: Readiness check
 *     description: Check if the service is ready to accept requests
 *     responses:
 *       200:
 *         description: Service is ready
 *       503:
 *         description: Service is not ready
 */
healthRouter.get('/ready', healthController.readinessCheck);

/**
 * @swagger
 * /health/live:
 *   get:
 *     summary: Liveness check
 *     description: Check if the service is running
 *     responses:
 *       200:
 *         description: Service is live
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: 'live'
 */
healthRouter.get('/live', healthController.livenessCheck);

/**
 * @swagger
 * /health/database:
 *   get:
 *     summary: Database health check
 *     description: Check database connection specifically
 *     responses:
 *       200:
 *         description: Database is healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: 'healthy'
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                   example: '2023-01-01T00:00:00.000Z'
 *       503:
 *         description: Database is unhealthy
 */
healthRouter.get('/database', async (_req, res) => {
  try {
    // Check database connection
    const isHealthy = await checkDatabaseConnection();
    
    res.status(isHealthy ? 200 : 503).json({
      status: isHealthy ? 'healthy' : 'unhealthy',
      message: isHealthy ? 'Database connection successful' : 'Database connection failed',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      message: 'Database health check failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
});
