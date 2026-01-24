import { Router } from 'express';
import { healthRouter } from './health';
import { apiRouter } from './api';
import { errorHandler } from '../../middleware/errorHandler';

const router = Router();

// API routes
router.use('/health', healthRouter);
router.use('/api', apiRouter);

// Error handling middleware - should be last
router.use(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  errorHandler as any
);

export { router as v1Router };
