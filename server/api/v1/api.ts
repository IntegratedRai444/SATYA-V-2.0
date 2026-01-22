import { Router } from 'express';
import { body } from 'express-validator';
import { v4 as uuidv4 } from 'uuid';
import { authenticate, requireRole } from '../../middleware/auth.middleware';
import { asyncHandler, validateRequest } from '../../middleware/errorHandler';
import { successResponse, errorResponse } from '../../utils/apiResponse';
import { logger } from '../../config/logger';
import { rateLimiter } from '../../middleware/rateLimiter';

const router = Router();

// Apply rate limiting to all API routes
router.use(rateLimiter);

// Apply authentication middleware to all API routes
router.use(authenticate);

// Health check endpoint
router.get('/health', asyncHandler(async (req, res) => {
  return successResponse(res, {
    status: 'ok',
    timestamp: new Date().toISOString(),
    requestId: req.id,
  });
}));

// User profile
router.get('/user/profile', asyncHandler(async (req, res) => {
  // In a real app, fetch user from database
  const user = {
    id: req.user?.id,
    email: req.user?.email,
    role: req.user?.role || 'user',
    user_metadata: req.user?.user_metadata || {},
  };
  
  return successResponse(res, { user });
}));

// Update user profile
router.put(
  '/user/profile',
  [
    body('email').optional().isEmail().withMessage('Invalid email'),
    body('name').optional().isString().trim().notEmpty(),
  ],
  validateRequest,
  asyncHandler(async (req, res) => {
    // In a real app, update user in database
    const updates = req.body;
    
    return successResponse(res, {
      message: 'Profile updated successfully',
      updates,
    });
  })
);

// File upload endpoint
router.post(
  '/analyze',
  // Add file validation middleware
  asyncHandler(async (req, res) => {
    // In a real app, process the file upload
    const file = req.file;
    
    if (!file) {
      return errorResponse(res, {
        code: 'NO_FILE_UPLOADED',
        message: 'No file was uploaded',
      }, 400);
    }

    // Create a job ID
    const jobId = uuidv4();
    
    // In a real app, queue the job for processing
    logger.info(`File upload received for job ${jobId}`, {
      originalname: file.originalname,
      mimetype: file.mimetype,
      size: file.size,
    });

    return successResponse(res, {
      jobId,
      status: 'queued',
      message: 'File uploaded and queued for processing',
    });
  })
);

// Get job status
router.get('/jobs/:jobId', asyncHandler(async (req, res) => {
  const { jobId } = req.params;
  
  // In a real app, fetch job status from database/queue
  const job = {
    id: jobId,
    status: 'completed', // or 'processing', 'failed', etc.
    progress: 100,
    result: null, // or the processed result
    createdAt: new Date().toISOString(),
  };

  return successResponse(res, { job });
}));

// Admin-only route example
router.get(
  '/admin/stats',
  requireRole(['admin']),
  asyncHandler(async (req, res) => {
    // In a real app, fetch admin stats
    const stats = {
      totalUsers: 100,
      activeUsers: 42,
      jobsProcessed: 1000,
    };
    
    return successResponse(res, { stats });
  })
);

export const apiRouter = router;
