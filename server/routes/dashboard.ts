import { Router, Response } from 'express';
import { z } from 'zod';
import { requireAuth, AuthenticatedRequest } from '../middleware/auth';
import { dashboardService } from '../services/dashboard-service';
import { logger } from '../config';

const router = Router();

// Validation schemas
const scanHistoryQuerySchema = z.object({
  limit: z.coerce.number().min(1).max(100).default(20),
  offset: z.coerce.number().min(0).default(0),
  type: z.enum(['image', 'video', 'audio']).optional(),
  result: z.enum(['authentic', 'deepfake', 'uncertain']).optional(),
  dateFrom: z.coerce.date().optional(),
  dateTo: z.coerce.date().optional(),
  sortBy: z.enum(['createdAt', 'confidenceScore', 'filename']).default('createdAt'),
  sortOrder: z.enum(['asc', 'desc']).default('desc')
});

/**
 * GET /api/dashboard/stats
 * Get dashboard statistics for the authenticated user
 */
router.get('/stats',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      const stats = await dashboardService.getDashboardStats(req.user.userId);

      logger.debug('Dashboard stats retrieved', {
        userId: req.user.userId,
        totalScans: stats.totalScans
      });

      res.json({
        success: true,
        message: 'Dashboard statistics retrieved successfully',
        data: stats
      });

    } catch (error) {
      logger.error('Failed to get dashboard stats', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve dashboard statistics'
      });
    }
  }
);

/**
 * GET /api/dashboard/analytics
 * Get detailed user analytics
 */
router.get('/analytics',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      const analytics = await dashboardService.getUserAnalytics(req.user.userId);

      logger.debug('User analytics retrieved', {
        userId: req.user.userId,
        totalScans: analytics.user.totalScans
      });

      res.json({
        success: true,
        message: 'User analytics retrieved successfully',
        data: analytics
      });

    } catch (error) {
      logger.error('Failed to get user analytics', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve user analytics'
      });
    }
  }
);

/**
 * GET /api/dashboard/scans
 * Get scan history with filtering and pagination
 */
router.get('/scans',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      // Validate query parameters
      const queryParams = scanHistoryQuerySchema.parse(req.query);

      const result = await dashboardService.getScanHistory(req.user.userId, queryParams);

      logger.debug('Scan history retrieved', {
        userId: req.user.userId,
        total: result.pagination.total,
        returned: result.scans.length
      });

      res.json({
        success: true,
        message: 'Scan history retrieved successfully',
        data: result.scans,
        pagination: result.pagination
      });

    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          message: 'Invalid query parameters',
          errors: error.errors.map(err => ({
            field: err.path.join('.'),
            message: err.message
          }))
        });
      }

      logger.error('Failed to get scan history', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve scan history'
      });
    }
  }
);

/**
 * GET /api/dashboard/scans/:scanId
 * Get detailed information about a specific scan
 */
router.get('/scans/:scanId',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      const scanId = parseInt(req.params.scanId);
      if (isNaN(scanId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid scan ID'
        });
      }

      const scan = await dashboardService.getScanDetails(scanId, req.user.userId);

      if (!scan) {
        return res.status(404).json({
          success: false,
          message: 'Scan not found'
        });
      }

      logger.debug('Scan details retrieved', {
        scanId,
        userId: req.user.userId
      });

      res.json({
        success: true,
        message: 'Scan details retrieved successfully',
        data: scan
      });

    } catch (error) {
      logger.error('Failed to get scan details', {
        error: (error as Error).message,
        scanId: req.params.scanId,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve scan details'
      });
    }
  }
);

/**
 * GET /api/dashboard/system-stats
 * Get system-wide statistics (admin only)
 */
router.get('/system-stats',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      // Check if user is admin
      if (req.user.role !== 'admin') {
        return res.status(403).json({
          success: false,
          message: 'Admin access required'
        });
      }

      const systemStats = await dashboardService.getSystemStats();

      logger.info('System stats retrieved by admin', {
        adminUserId: req.user.userId,
        totalUsers: systemStats.totalUsers,
        totalScans: systemStats.totalScans
      });

      res.json({
        success: true,
        message: 'System statistics retrieved successfully',
        data: systemStats
      });

    } catch (error) {
      logger.error('Failed to get system stats', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve system statistics'
      });
    }
  }
);

/**
 * POST /api/dashboard/save-scan
 * Save a scan result to the database (internal use)
 */
router.post('/save-scan',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      const { filename, type, result } = req.body;

      if (!filename || !type || !result) {
        return res.status(400).json({
          success: false,
          message: 'Missing required fields: filename, type, result'
        });
      }

      if (!['image', 'video', 'audio'].includes(type)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid file type. Must be image, video, or audio'
        });
      }

      const scanId = await dashboardService.saveScanResult(
        req.user.userId,
        filename,
        type,
        result
      );

      logger.info('Scan result saved via API', {
        scanId,
        userId: req.user.userId,
        filename,
        type
      });

      res.json({
        success: true,
        message: 'Scan result saved successfully',
        scanId
      });

    } catch (error) {
      logger.error('Failed to save scan result', {
        error: (error as Error).message,
        userId: req.user?.userId,
        body: req.body
      });

      res.status(500).json({
        success: false,
        message: 'Failed to save scan result'
      });
    }
  }
);

/**
 * GET /api/dashboard/recent-activity
 * Get recent activity summary
 */
router.get('/recent-activity',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      // Get recent scans (last 10)
      const recentScans = await dashboardService.getScanHistory(req.user.userId, {
        limit: 10,
        offset: 0,
        sortBy: 'createdAt',
        sortOrder: 'desc'
      });

      // Get basic stats for context
      const stats = await dashboardService.getDashboardStats(req.user.userId);

      const activitySummary = {
        recentScans: recentScans.scans.map(scan => ({
          id: scan.id,
          filename: scan.filename,
          type: scan.type,
          result: scan.result,
          confidenceScore: scan.confidenceScore,
          createdAt: scan.createdAt
        })),
        summary: {
          totalScans: stats.totalScans,
          scansLast7Days: stats.recentActivity.last7Days,
          scansLast30Days: stats.recentActivity.last30Days,
          averageConfidence: stats.averageConfidence
        }
      };

      res.json({
        success: true,
        message: 'Recent activity retrieved successfully',
        data: activitySummary
      });

    } catch (error) {
      logger.error('Failed to get recent activity', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve recent activity'
      });
    }
  }
);

export default router;