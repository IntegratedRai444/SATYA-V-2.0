import { Router, Request, type Response } from 'express';
import { supabaseAuth } from '../middleware/supabase-auth';
import { supabase } from '../config/supabase';
import rateLimit from 'express-rate-limit';
import { auditLogger } from '../middleware/audit-logger';
import '../types/express'; // Import to load Express type extensions

const router = Router();

// Rate limiter for notification endpoints
const notificationRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window per IP
  message: 'Too many notification requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// GET /api/v2/notifications - Get user notifications
router.get('/', notificationRateLimit, supabaseAuth, auditLogger('sensitive_data_access', 'notifications'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { limit = 20, offset = 0, unread_only = false } = req.query;

    let query = supabase
      .from('notifications')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .range(Number(offset), Number(offset) + Number(limit) - 1);

    if (unread_only === 'true') {
      query = query.eq('is_read', false);
    }

    const { data: notifications, error } = await query;

    if (error) {
      throw error;
    }

    // Get unread count
    const { count: unreadCount } = await supabase
      .from('notifications')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', userId)
      .eq('is_read', false);

    res.json({
      success: true,
      data: {
        notifications: notifications || [],
        unread_count: unreadCount || 0,
        pagination: {
          limit: Number(limit),
          offset: Number(offset),
          total: notifications?.length || 0
        }
      }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Get notifications error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// POST /api/v2/notifications - Create notification (for system use)
router.post('/', notificationRateLimit, supabaseAuth, auditLogger('notification_create', 'notification'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { type, title, message, action_url, action_label } = req.body;

    if (!type || !title || !message) {
      return res.status(400).json({
        success: false,
        error: 'Type, title, and message are required'
      });
    }

    const { data: notification, error } = await supabase
      .from('notifications')
      .insert({
        user_id: userId,
        type,
        title,
        message,
        action_url,
        action_label,
        is_read: false
      })
      .select()
      .single();

    if (error) {
      throw error;
    }

    res.status(201).json({
      success: true,
      data: notification
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Create notification error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// PUT /api/v2/notifications/:id/read - Mark notification as read
router.put('/:id/read', notificationRateLimit, supabaseAuth, auditLogger('notification_read', 'notification'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { data: notification, error } = await supabase
      .from('notifications')
      .update({ is_read: true })
      .eq('id', id)
      .eq('user_id', userId)
      .select()
      .single();

    if (error) {
      throw error;
    }

    if (!notification) {
      return res.status(404).json({
        success: false,
        error: 'Notification not found'
      });
    }

    res.json({
      success: true,
      data: notification
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Mark notification as read error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// PUT /api/v2/notifications/read-all - Mark all notifications as read
router.put('/read-all', notificationRateLimit, supabaseAuth, auditLogger('notification_read', 'notifications'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { error } = await supabase
      .from('notifications')
      .update({ is_read: true })
      .eq('user_id', userId)
      .eq('is_read', false);

    if (error) {
      throw error;
    }

    res.json({
      success: true,
      message: 'All notifications marked as read'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Mark all notifications as read error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// DELETE /api/v2/notifications/:id - Delete notification
router.delete('/:id', notificationRateLimit, supabaseAuth, auditLogger('admin_action', 'notification'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { error } = await supabase
      .from('notifications')
      .delete()
      .eq('id', id)
      .eq('user_id', userId);

    if (error) {
      throw error;
    }

    res.json({
      success: true,
      message: 'Notification deleted successfully'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Delete notification error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// DELETE /api/v2/notifications - Clear all notifications
router.delete('/', notificationRateLimit, supabaseAuth, auditLogger('admin_action', 'notifications'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { error } = await supabase
      .from('notifications')
      .delete()
      .eq('user_id', userId);

    if (error) {
      throw error;
    }

    res.json({
      success: true,
      message: 'All notifications cleared successfully'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Clear notifications error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/v2/notifications/unread-count - Get unread notification count
router.get('/unread-count', notificationRateLimit, supabaseAuth, auditLogger('sensitive_data_access', 'notifications'), async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    const { count } = await supabase
      .from('notifications')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', userId)
      .eq('is_read', false);

    res.json({
      success: true,
      data: {
        unread_count: count || 0
      }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Get unread count error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

export { router as notificationsRouter };
