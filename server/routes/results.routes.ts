import { Router, Request, Response } from 'express';
import { supabase, supabaseAdmin } from '../config/supabase';
import { logger } from '../config/logger';

const router = Router();

// GET /api/v2/results/:id - Get analysis result by ID
router.get('/:id', async (req: Request, res: Response) => {
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
      .select(`id, type, status, file_name, file_type, file_size, result, report_code, created_at, completed_at, error`)
      .eq('id', id)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .single();

    if (error || !task) {
      // Check if this is a stuck processing job (fail-safe)
      if (!task) {
        // Try to find the job without user filter for recovery
        const { data: orphanJob } = await supabase
          .from('tasks')
          .select('id, status, created_at')
          .eq('id', id)
          .eq('type', 'analysis')
          .single();
          
        if (orphanJob && orphanJob.status === 'processing') {
          // Job is stuck in processing - mark as failed
          await supabaseAdmin
            .from('tasks')
            .update({
              status: 'failed',
              error: 'Job failed due to processing timeout',
              completed_at: new Date().toISOString()
            })
            .eq('id', id);
            
          return res.status(200).json({
            id,
            type: 'analysis',
            status: 'failed',
            error: 'Job failed due to processing timeout',
            created_at: orphanJob.created_at,
            completed_at: new Date().toISOString()
          });
        }
      }
      
      return res.status(404).json({
        success: false,
        error: 'Analysis result not found'
      });
    }

    // 3-State Model: Handle processing jobs
    if (task.status === 'processing') {
      // Check if job is stuck (timeout recovery)
      const jobAge = Date.now() - new Date(task.created_at).getTime();
      const TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
      
      if (jobAge > TIMEOUT_MS) {
        // Job is stuck - mark as failed
        await supabaseAdmin
          .from('tasks')
          .update({
            status: 'failed',
            error: 'Job failed due to timeout',
            completed_at: new Date().toISOString()
          })
          .eq('id', id);
          
        return res.status(200).json({
          id: task.id,
          type: task.type,
          status: 'failed',
          error: 'Job failed due to timeout',
          created_at: task.created_at,
          completed_at: new Date().toISOString()
        });
      }
      
      return res.status(200).json({
        id: task.id,
        type: task.type,
        status: 'processing',
        file_name: task.file_name,
        file_type: task.file_type,
        file_size: task.file_size,
        created_at: task.created_at,
        message: 'Analysis is still in progress'
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
router.delete('/:id', async (req: Request, res: Response) => {
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
