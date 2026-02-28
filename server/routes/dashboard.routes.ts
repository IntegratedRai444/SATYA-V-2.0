import { Router, Response, Request } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import { requireAuth, authenticate } from '../middleware/auth.middleware';
import { parseAnalysisResult } from '../utils/result-parser';

// Centralized table constant to prevent schema drift
const ANALYSIS_TABLE = 'tasks';

const router = Router();

// 🔥 PRODUCTION: Development-friendly auth check
const isDevelopment = process.env.NODE_ENV === 'development';

// Rate limiter for dashboard endpoints
const dashboardRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window per IP
  message: 'Too many dashboard requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// 🔥 PRODUCTION: Enhanced auth middleware with development fallback
const checkAuth = (req: Request, res: Response, next: (err?: unknown) => void) => {
  // In development, allow requests without auth for easier testing
  if (isDevelopment) {
    logger.info('[DASHBOARD] Development mode - auth bypassed');
    return next();
  }
  
  // In production, require auth
  return authenticate(req, res, next);
};

// GET /api/v2/dashboard/stats - Get dashboard statistics
// eslint-disable-next-line @typescript-eslint/no-explicit-any
router.get('/stats', dashboardRateLimit, checkAuth, async (req: Request, res: Response) => {
  try {
    // 🔥 PRODUCTION: Enhanced user detection with fallback
    let userId = req.user?.id;
    
    // Try to get user from Supabase auth if not in req.user
    if (!userId) {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        userId = user?.id;
        logger.info('[DASHBOARD] User retrieved from Supabase auth', { userId });
      } catch (authError) {
        logger.warn('[DASHBOARD] No user found in request or Supabase auth', {
          authError: authError instanceof Error ? authError.message : 'Unknown error'
        });
      }
    }
    
    // 🔥 PRODUCTION: Development-friendly auth handling
    if (!userId && !isDevelopment) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized - No valid user session found',
        devMode: isDevelopment
      });
    }
    
    // 🔥 PRODUCTION: Log user access for debugging
    logger.info('[DASHBOARD_STATS] User access', {
      userId: userId || 'development-user',
      devMode: isDevelopment,
      hasUserInReq: !!req.user,
      hasUserInAuth: !!userId
    });

    // 🔥 PRODUCTION: Null-safe database query with proper error handling
    let tasks: Array<{
      id: string;
      result: unknown;
      type: string;
      created_at: string;
      confidence_score: number;
      filename?: string;
    }> = [];
    try {
      const { data: tasksData, error: tasksError } = await supabase
        .from(ANALYSIS_TABLE)
        .select('id, result, type, created_at, confidence_score, filename')
        .eq('user_id', userId)
        .eq('type', 'analysis')
        .is('deleted_at', null) // 🔥 CRITICAL: Add soft delete filter
        .order('created_at', { ascending: false })
        .limit(1000);

      if (tasksError) {
        logger.error('[DASHBOARD_STATS] Database query failed', {
          userId,
          error: tasksError.message,
          code: tasksError.code
        });
        // Return safe fallback instead of throwing
        tasks = [];
      }

      tasks = tasksData || [];
    } catch (dbError) {
      logger.error('[DASHBOARD_STATS] Unexpected database error', {
        userId,
        error: dbError instanceof Error ? dbError.message : 'Unknown error'
      });
      // Return safe fallback instead of throwing
      tasks = [];
    }

    // Calculate statistics with safe defaults and null safety
    const totalAnalyses = tasks?.length || 0;
    
    // Safely parse results to prevent crashes
    const parsedResults = tasks?.map(task => {
      try {
        return parseAnalysisResult(task.result);
      } catch (error) {
        logger.warn('Failed to parse analysis result', { 
          taskId: task.id, 
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        return { prediction: 'inconclusive', confidence: 0, reasoning: 'Parse error' };
      }
    }) || [];
    
    const deepfakeDetected = parsedResults.filter(r => r.prediction === 'fake').length;
    const realDetected = parsedResults.filter(r => r.prediction === 'real').length;
    
    // Calculate average confidence with safe defaults
    const validConfidences = parsedResults.filter(r => r.confidence > 0);
    const avgConfidence = validConfidences.length > 0 
      ? validConfidences.reduce((sum, r) => sum + r.confidence, 0) / validConfidences.length 
      : 0;

    // Get last 7 days activity with safe defaults
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    const last7Days = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      const dayCount = tasks?.filter((task: { created_at?: string }) => {
        const taskDate = new Date(task.created_at || '').toISOString().split('T')[0];
        return taskDate === dateStr;
      }).length || 0;
      
      last7Days.push({
        date: dateStr,
        count: dayCount
      });
    }

    const stats = {
      totalAnalyses: totalAnalyses || 0,
      authenticMedia: realDetected || 0,
      manipulatedMedia: deepfakeDetected || 0,
      uncertainAnalyses: 0, // Calculate if needed
      avgConfidence: Math.round(avgConfidence * 100) / 100,
      last_7_days: last7Days || []
    };

    res.json({
      success: true,
      data: stats
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch dashboard statistics';
    logger.error('Dashboard stats error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/v2/dashboard/analytics - Get user analytics (alias for user/analytics)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
router.get('/analytics', dashboardRateLimit, async (req: any, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Get user's analysis data
    const { data: tasks, error: tasksError } = await supabase
      .from(ANALYSIS_TABLE)
      .select('id, type, created_at, filename, result, confidence_score')
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .is('deleted_at', null) // 🔥 CRITICAL: Add soft delete filter
      .order('created_at', { ascending: false })
      .limit(100); // Recent 50 activities

    if (tasksError) {
      logger.error('Dashboard analytics error:', tasksError);
      return res.status(500).json({
        success: false,
        error: 'Failed to fetch user analytics'
      });
    }

    // Calculate usage by type
    const usageByType = {
      image: tasks?.filter(t => t.type === 'image').length || 0,
      audio: tasks?.filter(t => t.type === 'audio').length || 0,
      video: tasks?.filter(t => t.type === 'video').length || 0,
      multimodal: tasks?.filter(t => t.type === 'multimodal').length || 0,
      webcam: tasks?.filter(t => t.type === 'webcam').length || 0
    };

    // Format recent activity with safe parsing
    const recentActivity = tasks?.map(task => {
      try {
        const parsedResult = parseAnalysisResult(task.result);
        return {
          id: task.id,
          modality: task.type,
          created_at: task.created_at,
          confidence: parsedResult.confidence,
          is_deepfake: parsedResult.prediction === 'fake',
          prediction: parsedResult.prediction,
          reasoning: parsedResult.reasoning,
          file_name: task.filename || 'Unknown'
        };
      } catch (error) {
        logger.warn('Failed to format activity item', { 
          taskId: task.id, 
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        return {
          id: task.id,
          modality: task.type,
          created_at: task.created_at,
          confidence: 0,
          is_deepfake: false,
          prediction: 'inconclusive',
          reasoning: 'Parse error',
          file_name: task.filename || 'Unknown'
        };
      }
    }) || [];

    const analytics = {
      usage_by_type: usageByType,
      recent_activity: recentActivity
    };

    res.json({
      success: true,
      analytics
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch user analytics';
    logger.error('Dashboard analytics error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/v2/dashboard/recent-activity - Get recent user activity
// eslint-disable-next-line @typescript-eslint/no-explicit-any
router.get('/recent-activity', requireAuth, dashboardRateLimit, async (req: any, res: Response) => {
  try {
    const userId = req.user?.id;
    
    // Get recent analysis activities (last 10)
    const { data: activities, error: activitiesError } = await supabase
      .from(ANALYSIS_TABLE)
      .select('id, type, result, created_at, confidence_score, filename')
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .is('deleted_at', null) // 🔥 CRITICAL: Add soft delete filter
      .order('created_at', { ascending: false })
      .limit(10);

    if (activitiesError) {
      logger.error('[DB] Recent activity query failed:', activitiesError);
      return res.status(500).json({
        success: false,
        error: 'Database query failed'
      });
    }

    res.json({
      success: true,
      data: activities || []
    });
  } catch (error) {
    logger.error('[ERROR] Recent activity endpoint error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

export { router as dashboardRouter };
