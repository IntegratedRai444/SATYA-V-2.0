import { Router } from 'express';
import authRoutes from './auth';
import sessionRoutes from './session';
import uploadRoutes from './upload';
import processingRoutes from './processing';
import analysisRoutes from './analysis';
import dashboardRoutes from './dashboard';
import healthRoutes from './health';

const router = Router();

// Register all route modules
router.use('/auth', authRoutes);
router.use('/session', sessionRoutes);
router.use('/upload', uploadRoutes);
router.use('/processing', processingRoutes);
router.use('/analysis', analysisRoutes);
router.use('/dashboard', dashboardRoutes);
router.use('/health', healthRoutes);

export default router;