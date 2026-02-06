import { Router, Request, type Response } from 'express';
import { body, validationResult } from 'express-validator';
import { supabase } from '../config/supabase';
import rateLimit from 'express-rate-limit';
import { auditLogger } from '../middleware/audit-logger';
import { logger } from '../config/logger';

const router = Router();

// Rate limiter for user profile endpoints
const userRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 50, // 50 requests per window per IP
  message: 'Too many user profile requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// GET /api/v2/user/profile - Get user profile
router.get('/profile', userRateLimit, auditLogger('sensitive_data_access', 'user_profile'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Get user profile from users table
    const { data: userProfile, error: profileError } = await supabase
      .from('users')
      .select('id, username, email, full_name, avatar_url, role, is_active, last_login, created_at, updated_at')
      .eq('id', userId)
      .single();

    if (profileError) {
      throw profileError;
    }

    // Get user preferences
    const { data: preferences, error: preferencesError } = await supabase
      .from('user_preferences')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (preferencesError && preferencesError.code !== 'PGRST116') {
      throw preferencesError;
    }

    res.json({
      success: true,
      data: {
        profile: userProfile,
        preferences: preferences || {
          theme: 'dark',
          language: 'english',
          confidence_threshold: 75,
          enable_notifications: true,
          auto_analyze: true,
          sensitivity_level: 'medium',
          chat_model: 'gpt-4'
        }
      }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch user profile';
    logger.error('Get profile error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// PUT /api/v2/user/profile - Update user profile
router.put('/profile', [
  body('fullName').optional().isString().isLength({ min: 1, max: 100 }),
  body('username').optional().isString().isLength({ min: 3, max: 30 }).matches(/^[a-zA-Z0-9_]+$/),
  body('avatar_url').optional().isURL(),
], userRateLimit, auditLogger('user_profile_update', 'user_profile'), async (req: Request, res: Response) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { fullName, username, avatar_url } = req.body;

    // Update user profile
    const updateData: Record<string, unknown> = {};
    if (fullName !== undefined) updateData.full_name = fullName;
    if (username !== undefined) updateData.username = username;
    if (avatar_url !== undefined) updateData.avatar_url = avatar_url;
    updateData.updated_at = new Date().toISOString();

    const { data: updatedProfile, error: profileError } = await supabase
      .from('users')
      .update(updateData)
      .eq('id', userId)
      .select('id, username, email, full_name, avatar_url, role, is_active, last_login, created_at, updated_at')
      .single();

    if (profileError) {
      throw profileError;
    }

    res.json({
      success: true,
      data: updatedProfile,
      message: 'Profile updated successfully'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to update user profile';
    logger.error('Update profile error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// PUT /api/v2/user/preferences - Update user preferences
router.put('/preferences', [
  body('theme').optional().isIn(['light', 'dark', 'auto']),
  body('language').optional().isString().isLength({ min: 2, max: 10 }),
  body('confidence_threshold').optional().isInt({ min: 0, max: 100 }),
  body('enable_notifications').optional().isBoolean(),
  body('auto_analyze').optional().isBoolean(),
  body('sensitivity_level').optional().isIn(['low', 'medium', 'high']),
  body('chat_model').optional().isString().isLength({ min: 1, max: 50 }),
], userRateLimit, auditLogger('user_preferences_update', 'user_preferences'), async (req: Request, res: Response) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const preferences = req.body;

    // Update or insert user preferences
    const { data: updatedPreferences, error: preferencesError } = await supabase
      .from('user_preferences')
      .upsert({
        user_id: userId,
        ...preferences,
        updated_at: new Date().toISOString()
      })
      .select('*')
      .single();

    if (preferencesError) {
      throw preferencesError;
    }

    res.json({
      success: true,
      data: updatedPreferences,
      message: 'Preferences updated successfully'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to update user preferences';
    logger.error('Update preferences error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// DELETE /api/v2/user/account - Delete user account (soft delete)
router.delete('/account', userRateLimit, auditLogger('user_delete', 'user_account'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Soft delete user by setting deleted_at
    const { error: userError } = await supabase
      .from('users')
      .update({ deleted_at: new Date().toISOString() })
      .eq('id', userId);

    if (userError) {
      throw userError;
    }

    // Soft delete user preferences
    await supabase
      .from('user_preferences')
      .update({ deleted_at: new Date().toISOString() })
      .eq('user_id', userId);

    // Soft delete user's tasks
    await supabase
      .from('tasks')
      .update({ deleted_at: new Date().toISOString() })
      .eq('user_id', userId);

    // Soft delete user's notifications
    await supabase
      .from('notifications')
      .update({ deleted_at: new Date().toISOString() })
      .eq('user_id', userId);

    // Soft delete user's chat conversations
    await supabase
      .from('chat_conversations')
      .update({ deleted_at: new Date().toISOString() })
      .eq('user_id', userId);

    res.json({
      success: true,
      message: 'Account deleted successfully'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to delete user account';
    logger.error('Delete account error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/v2/user/stats - Get user statistics
router.get('/stats', userRateLimit, auditLogger('sensitive_data_access', 'user_statistics'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Get user's analysis statistics
    const { data: tasks } = await supabase
      .from('tasks')
      .select('status, type, created_at')
      .eq('user_id', userId)
      .eq('deleted_at', null);

    const stats = {
      total_analyses: tasks?.length || 0,
      completed_analyses: tasks?.filter(t => t.status === 'completed').length || 0,
      pending_analyses: tasks?.filter(t => ['pending', 'queued', 'processing'].includes(t.status)).length || 0,
      failed_analyses: tasks?.filter(t => t.status === 'failed').length || 0,
      image_analyses: tasks?.filter(t => t.type === 'image').length || 0,
      video_analyses: tasks?.filter(t => t.type === 'video').length || 0,
      audio_analyses: tasks?.filter(t => t.type === 'audio').length || 0,
      total_chats: 0,
      unread_notifications: 0
    };

    res.json({
      success: true,
      data: stats
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch user statistics';
    logger.error('Get user stats error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/v2/user/analytics - Get user analytics
router.get('/analytics', userRateLimit, auditLogger('sensitive_data_access', 'user_analytics'), async (req: Request, res: Response) => {
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
      .select('id, type, created_at, file_name, result')
      .eq('user_id', userId)
      .eq('deleted_at', null)
      .order('created_at', { ascending: false })
      .limit(50); // Recent 50 activities

    if (tasksError) {
      logger.error('User analytics error:', {
        error: tasksError,
        userId: req.user?.id
      });
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
      confidence: (task.result as Record<string, unknown>)?.confidence as number || 0,
      is_deepfake: (task.result as Record<string, unknown>)?.is_deepfake as boolean || false,
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
    logger.error('Get user analytics error:', {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      userId: req.user?.id
    });
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

export { router as userRouter };
