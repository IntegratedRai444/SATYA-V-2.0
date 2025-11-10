import { Router, Response } from 'express';
import { sessionManager } from '../services/session-manager';
import { jwtAuthService } from '../services/jwt-auth-service';
import { requireAuth, AuthenticatedRequest } from '../middleware/auth';
import { logger, logSecurity } from '../config';

const router = Router();

/**
 * GET /api/session/current
 * Get current session information
 */
router.get('/current', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    const session = sessionManager.getSession(token);
    if (!session) {
      return res.status(404).json({
        success: false,
        message: 'Session not found'
      });
    }

    // Update session activity
    sessionManager.updateSessionActivity(token);

    res.json({
      success: true,
      session: {
        userId: session.userId,
        username: session.username,
        email: session.email,
        role: session.role,
        loginTime: session.loginTime,
        lastActivity: session.lastActivity,
        isActive: session.isActive
      }
    });
  } catch (error) {
    logger.error('Failed to get current session', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to retrieve session information'
    });
  }
});

/**
 * GET /api/session/all
 * Get all active sessions for current user
 */
router.get('/all', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
    }

    const sessions = sessionManager.getUserSessions(req.user.userId);
    const currentToken = req.headers.authorization?.replace('Bearer ', '');

    const sessionList = sessions.map(session => ({
      loginTime: session.loginTime,
      lastActivity: session.lastActivity,
      ipAddress: session.ipAddress,
      userAgent: session.userAgent,
      isActive: session.isActive,
      isCurrent: currentToken ? sessionManager.getSession(currentToken)?.loginTime === session.loginTime : false
    }));

    res.json({
      success: true,
      sessions: sessionList,
      totalSessions: sessions.length
    });
  } catch (error) {
    logger.error('Failed to get user sessions', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to retrieve sessions'
    });
  }
});

/**
 * DELETE /api/session/current
 * Logout from current session
 */
router.delete('/current', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    const success = await sessionManager.destroySession(token);
    if (success) {
      res.json({
        success: true,
        message: 'Session terminated successfully'
      });
    } else {
      res.status(404).json({
        success: false,
        message: 'Session not found'
      });
    }
  } catch (error) {
    logger.error('Failed to terminate current session', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to terminate session'
    });
  }
});

/**
 * DELETE /api/session/all
 * Logout from all sessions except current
 */
router.delete('/all', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
    }

    const currentToken = req.headers.authorization?.replace('Bearer ', '');
    const destroyedCount = await sessionManager.destroyAllUserSessions(
      req.user.userId,
      currentToken
    );

    logSecurity('User terminated all other sessions', {
      userId: req.user.userId,
      username: req.user.username,
      destroyedCount
    });

    res.json({
      success: true,
      message: `${destroyedCount} sessions terminated successfully`,
      destroyedCount
    });
  } catch (error) {
    logger.error('Failed to terminate all sessions', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to terminate sessions'
    });
  }
});

/**
 * DELETE /api/session/all-including-current
 * Logout from all sessions including current
 */
router.delete('/all-including-current', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
    }

    const destroyedCount = await sessionManager.destroyAllUserSessions(req.user.userId);

    logSecurity('User terminated all sessions including current', {
      userId: req.user.userId,
      username: req.user.username,
      destroyedCount
    });

    res.json({
      success: true,
      message: `All ${destroyedCount} sessions terminated successfully`,
      destroyedCount
    });
  } catch (error) {
    logger.error('Failed to terminate all sessions including current', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to terminate sessions'
    });
  }
});

/**
 * POST /api/session/refresh
 * Refresh current session (update activity)
 */
router.post('/refresh', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token is still valid
    const payload = await jwtAuthService.verifyToken(token);
    if (!payload) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    // Update session activity
    sessionManager.updateSessionActivity(token);

    res.json({
      success: true,
      message: 'Session refreshed successfully',
      user: {
        id: payload.userId,
        username: payload.username,
        email: payload.email,
        role: payload.role
      }
    });
  } catch (error) {
    logger.error('Failed to refresh session', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to refresh session'
    });
  }
});

/**
 * GET /api/session/stats
 * Get session statistics (admin only)
 */
router.get('/stats', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Check if user is admin
    if (req.user?.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Admin access required'
      });
    }

    const stats = sessionManager.getSessionStats();
    const activityReport = sessionManager.getActivityReport();

    res.json({
      success: true,
      stats: {
        totalActiveSessions: stats.totalActiveSessions,
        activeUsers: activityReport.activeUsers,
        averageSessionsPerUser: Math.round(activityReport.averageSessionsPerUser * 100) / 100,
        oldestSession: activityReport.oldestSession,
        newestSession: activityReport.newestSession
      }
    });
  } catch (error) {
    logger.error('Failed to get session stats', {
      error: (error as Error).message,
      userId: req.user?.userId
    });

    res.status(500).json({
      success: false,
      message: 'Failed to retrieve session statistics'
    });
  }
});

export default router;