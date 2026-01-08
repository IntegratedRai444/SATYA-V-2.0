import { Router } from 'express';
import { healthRouter } from './health';
import { authRouter } from '../../routes/auth.routes';
import { apiRouter } from './api';
import { errorHandler } from '../../middleware/errorHandler';

const router = Router();

// API routes
router.use('/health', healthRouter);
router.use('/auth', authRouter);
router.use('/api', apiRouter);

// Error handling middleware - should be last
router.use(errorHandler);

export { router as v1Router };
