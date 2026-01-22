import { Router, Request, Response } from 'express';
import { supabaseAuth } from '../middleware/supabase-auth';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

const router = Router();

// GET /api/v2/results/:id - Get analysis result by ID
router.get('/:id', supabaseAuth, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Fetch analysis result from tasks table
    const { data: task, error } = await supabase
      .from('tasks')
      .select(`
        id,
        type,
        status,
        file_name,
        file_type,
        file_size,
        result,
        report_code,
        created_at,
        completed_at,
        error
      `)
      .eq('id', id)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .single();

    if (error || !task) {
      return res.status(404).json({
        success: false,
        error: 'Analysis result not found'
      });
    }

    // Format the response
    const result = {
      id: task.id,
      type: task.type,
      status: task.status,
      filename: task.file_name,
      fileType: task.file_type,
      fileSize: task.file_size,
      reportCode: task.report_code,
      result: task.result,
      createdAt: task.created_at,
      completedAt: task.completed_at,
      error: task.error
    };

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    logger.error('Error fetching analysis result:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

// DELETE /api/v2/results/:id - Delete analysis result
router.delete('/:id', supabaseAuth, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Check if task belongs to user
    const { data: task, error: checkError } = await supabase
      .from('tasks')
      .select('id')
      .eq('id', id)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .single();

    if (checkError || !task) {
      return res.status(404).json({
        success: false,
        error: 'Analysis result not found'
      });
    }

    // Delete the task
    const { error: deleteError } = await supabase
      .from('tasks')
      .delete()
      .eq('id', id)
      .eq('user_id', userId);

    if (deleteError) {
      logger.error('Failed to delete analysis result:', deleteError);
      return res.status(500).json({
        success: false,
        error: 'Failed to delete analysis result'
      });
    }

    res.json({
      success: true,
      message: 'Analysis result deleted successfully'
    });
  } catch (error) {
    logger.error('Error deleting analysis result:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

export { router as resultsRouter };
