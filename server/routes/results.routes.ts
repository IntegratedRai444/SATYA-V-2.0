import { Router, Request, Response } from 'express';
import { supabase, supabaseAdmin } from '../config/supabase';
import { logger } from '../config/logger';
import { JOB_TIMEOUTS } from '../config/job-timeouts';

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

    // Fetch analysis result from scans table
    const { data: task, error } = await supabase
      .from('scans')
      .select(`id, type, filename, result, confidence_score, detection_details, metadata, created_at, updated_at`)
      .eq('id', id)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .is('deleted_at', null) // 🔥 CRITICAL FIX: Add soft delete filter like other routes
      .single();

    if (error || !task) {
      // 🔥 STEP 7 — ADD RACE CONDITION GUARD
      // Check if this is a race condition - job might still be inserting
      if (!task) {
        // Try to find job without user filter to check if it exists
        const { data: orphanJob } = await supabase
          .from('scans')
          .select('id, result, created_at')
          .eq('id', id)
          .eq('type', 'analysis')
          .single();
          
        if (orphanJob && orphanJob.result === 'processing') {
          // Job exists but user_id not yet committed - return processing state
          return res.status(200).json({
            success: true,
            jobId: orphanJob.id,
            status: 'processing',
            result: null
          });
        }
      }
      
      return res.status(404).json({
        success: false,
        error: 'Analysis result not found'
      });
    }

    // 3-State Model: Handle processing jobs
    if (task.result === 'processing') {
      // Check if job is stuck (timeout recovery)
      const jobAge = Date.now() - new Date(task.created_at).getTime();
      
      if (jobAge > JOB_TIMEOUTS.MAX_PROCESSING_TIME) {
        // 🔒 VERIFY OWNERSHIP BEFORE ADMIN UPDATE
        const { data: ownerCheck } = await supabase
          .from('scans')
          .select('user_id')
          .eq('id', id)
          .single();
          
        if (!ownerCheck || ownerCheck.user_id !== userId) {
          return res.status(403).json({ error: 'Access denied' });
        }
        
        // Job is stuck - mark as failed
        await supabaseAdmin
          .from('scans')
          .update({
            result: 'failed',
            confidence_score: 0,
            updated_at: new Date().toISOString()
          })
          .eq('id', id);
          
        return res.status(200).json({
          success: true,
          jobId: task.id,
          status: 'failed',
          result: null
        });
      }
      
      return res.status(200).json({
        success: true,
        jobId: task.id,
        status: 'processing',
        result: null
      });
    }

    // 🔥 STEP 1 — LOCK RESPONSE CONTRACT (CRITICAL)
    // Canonical response shape - NO nested wrappers, NO extra fields
    const result = {
      jobId: task.id,
      status: task.result,
      result: task.detection_details ?? null
    };

    logger.info('[RESULT_FETCH]', { 
      jobId: task.id, 
      status: task.result,
      hasResult: !!task.detection_details 
    });

    res.status(200).json({
      success: true,
      ...result // 🔥 FLATTENED RESPONSE - NO {data: {...}} wrapper
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
      .from('scans')
      .select('id')
      .eq('id', id)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .is('deleted_at', null) // 🔥 CRITICAL FIX: Add soft delete filter for consistency
      .single();

    if (checkError || !task) {
      return res.status(404).json({
        success: false,
        error: 'Analysis result not found'
      });
    }

    // Delete the task
    const { error: deleteError } = await supabase
      .from('scans')
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

    res.status(200).json({
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
