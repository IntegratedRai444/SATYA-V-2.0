import { Router, Request, Response } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';

const router = Router();

// Rate limiter for dashboard endpoints
const dashboardRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window per IP
  message: 'Too many dashboard requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// GET /api/v2/dashboard/stats - Get dashboard statistics
router.get('/stats', dashboardRateLimit, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Get user's analysis statistics (limited to last 1000 for performance)
    const { data: tasks, error: tasksError } = await supabase
      .from('tasks')
      .select('status, type, created_at, confidence, is_deepfake')
      .eq('user_id', userId)
      .eq('deleted_at', null)
      .order('created_at', { ascending: false })
      .limit(1000);

    if (tasksError) {
      logger.error('Dashboard stats error:', tasksError);
      return res.status(500).json({
        success: false,
        error: 'Failed to fetch dashboard statistics'
      });
    }

    // Calculate statistics
    const totalAnalyses = tasks?.length || 0;
    const deepfakeDetected = tasks?.filter(t => t.is_deepfake === true).length || 0;
    const realDetected = tasks?.filter(t => t.is_deepfake === false).length || 0;
    
    // Calculate average confidence
    const completedTasks = tasks?.filter(t => t.status === 'completed' && t.confidence) || [];
    const avgConfidence = completedTasks.length > 0 
      ? completedTasks.reduce((sum, task) => sum + (task.confidence || 0), 0) / completedTasks.length 
      : 0;

    // Get last 7 days activity
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    const last7Days = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      const dayCount = tasks?.filter(task => {
        const taskDate = new Date(task.created_at).toISOString().split('T')[0];
        return taskDate === dateStr;
      }).length || 0;
      
      last7Days.push({
        date: dateStr,
        count: dayCount
      });
    }

    const stats = {
      total_analyses: totalAnalyses,
      deepfake_detected: deepfakeDetected,
      real_detected: realDetected,
      avg_confidence: Math.round(avgConfidence * 100) / 100,
      last_7_days: last7Days
    };

    res.json({
      success: true,
      stats
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
router.get('/analytics', dashboardRateLimit, async (req: Request, res: Response) => {
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
      .from('tasks')
      .select('id, type, created_at, confidence, is_deepfake, file_name')
      .eq('user_id', userId)
      .eq('deleted_at', null)
      .order('created_at', { ascending: false })
      .limit(50); // Recent 50 activities

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

    // Format recent activity
    const recentActivity = tasks?.map(task => ({
      id: task.id,
      modality: task.type,
      created_at: task.created_at,
      confidence: task.confidence || 0,
      is_deepfake: task.is_deepfake || false,
      file_name: task.file_name || 'Unknown'
    })) || [];

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

export { router as dashboardRouter };
