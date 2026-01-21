import { Router } from 'express';
import { healthController } from '../controllers/health.controller';

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

export default healthRouter;
