import { Router, Request, Response } from 'express';
import { supabase, supabaseAdmin } from '../config/supabase';
import { logger } from '../config/logger';
import { randomUUID } from 'node:crypto';

const router = Router();


// GET /api/v2/history - Get paginated list of user's analysis jobs
router.get('/',
  async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'User not authenticated'
        });
      }

      // Pagination parameters
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 20;
      const offset = (page - 1) * limit;

      // Fetch user's analysis jobs with pagination
      const { data: jobs, error } = await supabase
        .from('tasks')
        .select(`
          id,
          type,
          filename,
          result,
          confidence_score,
          detection_details,
          metadata,
          created_at,
          updated_at
        `)
        .eq('user_id', userId)
        .eq('type', 'analysis')
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      if (error) {
        logger.error('Failed to fetch analysis history:', error);
        return res.status(500).json({
          success: false,
          error: 'Failed to fetch analysis history'
        });
      }

      // Get total count for pagination
      const { count } = await supabase
        .from('scans')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', userId)
        .eq('type', 'analysis')
        .is('deleted_at', null);

      const totalPages = Math.ceil((count || 0) / limit);

      res.json({
        success: true,
        data: {
          jobs: jobs?.map((job: {
            id: string;
            type: string;
            filename: string;
            result: string;
            confidence_score: number;
            detection_details: string;
            metadata: Record<string, unknown>;
            created_at: string;
            updated_at: string;
          }) => ({
            ...job,
            modality: job.type,
            filename: job.filename,
            confidence: job.confidence_score,
            detectionDetails: job.detection_details,
            metadata: job.metadata,
            reportCode: job.filename, // ADD THIS LINE
          })) || [],
          pagination: {
            page,
            limit,
            total: count || 0,
            totalPages,
            hasNext: page < totalPages,
            hasPrev: page > 1
          }
        }
      });
    } catch (error) {
      logger.error('History route error:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  }
);

// 🔥 STEP 5 — REMOVE LEGACY REDIRECT
// GET /api/v2/history/:jobId - REDIRECT to canonical results endpoint
router.get('/:jobId', async (req: Request, res: Response) => {
  try {
    const { jobId } = req.params;
    
    // 🔥 PURE REDIRECT - No DB queries inside legacy routes
    return res.redirect(307, `/api/v2/results/${jobId}`);
  } catch (error) {
    logger.error('History redirect error:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

// DELETE /api/v2/history/:jobId - Delete a specific analysis job
router.delete('/:jobId',
  async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const { jobId } = req.params;

      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'User not authenticated'
        });
      }

      // First check if job belongs to user
      const { data: job, error: checkError } = await supabase
        .from('scans')
        .select('id')
        .eq('id', jobId)
        .eq('user_id', userId)
        .eq('type', 'analysis')
        .single();

      if (checkError || !job) {
        return res.status(404).json({
          success: false,
          error: 'Analysis job not found'
        });
      }

      // Soft delete the job (set deleted_at timestamp)
      const { error: deleteError } = await supabase
        .from('scans')
        .update({ deleted_at: new Date().toISOString() })
        .eq('id', jobId)
        .eq('user_id', userId)
        .eq('type', 'analysis');

      if (deleteError) {
        logger.error('Failed to delete analysis job:', deleteError);
        return res.status(500).json({
          success: false,
          error: 'Failed to delete analysis job'
        });
      }

      res.json({
        success: true,
        message: 'Analysis job deleted successfully'
      });
    } catch (error) {
      logger.error('History delete route error:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  }
);

// Helper function to create analysis job (used by analysis routes)
export const createAnalysisJob = async (
  userId: string,
  jobData: {
    modality: 'image' | 'audio' | 'video' | 'text';
    filename: string;
    mime_type: string;
    size_bytes: number;
    metadata?: Record<string, unknown>;
    status?: 'pending' | 'processing' | 'completed' | 'failed';
  },
  jobId?: string // Allow external jobId to be passed
) => {
  // Generate jobId if not provided (ensures consistency)
  const finalJobId = jobId || randomUUID();
  
  const { data, error } = await supabaseAdmin
    .from('tasks')
    .insert({
      id: finalJobId, // Use consistent ID
      user_id: userId,
      type: jobData.modality,
      filename: jobData.filename,
      result: 'processing',
      confidence_score: 0,
      detection_details: null,
      metadata: {
        ...jobData.metadata,
        media_type: jobData.mime_type,
        file_size: jobData.size_bytes
      },
      created_at: new Date().toISOString()
    })
    .select('id')
    .single();

  if (error) {
    logger.error('TASK INSERT FAILED', { 
      jobId: finalJobId, 
      error: error.message,
      details: error,
      userId,
      jobData 
    });
    throw new Error(`Failed to create analysis job: ${error.message}`);
  }

  logger.info('TASK INSERT SUCCESS', { 
    jobId: finalJobId,
    returnedId: data?.id
  });

  return data;
};

export const getPaginatedAnalysisHistory = async (
  userId: string,
  page: number = 1,
  limit: number = 20
) => {
  const offset = (page - 1) * limit;

  const { data, error } = await supabase
    .from('scans')
    .select(`
      id,
      type,
      filename,
      result,
      confidence_score,
      detection_details,
      metadata,
      created_at,
      updated_at
    `)
    .eq('user_id', userId)
    .eq('type', 'analysis')
    .order('created_at', { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) {
    logger.error('Failed to fetch analysis history:', error);
    return { items: [], total: 0 };
  }

  const { count } = await supabase
    .from('scans')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', userId)
    .eq('type', 'analysis');

  return { items: data || [], total: count || 0 };
};

export const updateAnalysisJobWithResults = async (
  jobId: string,
  results: {
    status: 'completed' | 'failed';
    confidence?: number;
    is_deepfake?: boolean;
    model_name?: string;
    model_version?: string;
    summary?: Record<string, unknown>;
    analysis_data?: Record<string, unknown>;
    proof_json?: Record<string, unknown>;
    error_message?: string;
  }
) => {
  // 🔥 FINALIZATION GUARD: Check current status to prevent duplicate updates
  const { data: currentJob } = await supabaseAdmin
    .from('tasks')
    .select('result, updated_at')
    .eq('id', jobId)
    .single();

  // Skip update if already in final state
  if (currentJob && (currentJob.result === 'completed' || currentJob.result === 'failed')) {
    logger.warn('[FINALIZATION GUARD] Skipping duplicate update', {
      jobId,
      currentStatus: currentJob.result,
      attemptedStatus: results.status,
      lastUpdated: currentJob.updated_at
    });
    return currentJob;
  }

  const { data, error: jobUpdateError } = await supabaseAdmin
    .from('tasks')
    .update({
      result: results.status,
      confidence_score: results.confidence || (results.is_deepfake ? 0.8 : 0.2),
      detection_details: {
        confidence: results.confidence || 0,
        is_deepfake: results.is_deepfake || false,
        model_name: results.model_name || 'SatyaAI',
        model_version: results.model_version || '1.0.0',
        summary: results.summary || {},
        analysis_data: results.analysis_data || {},
        proof_json: results.proof_json || {},
        error_message: results.error_message
      },
      updated_at: new Date().toISOString()
    })
    .eq('id', jobId)
    .select('id, result, updated_at')
    .single();

  if (jobUpdateError) {
    logger.error('Failed to update analysis job:', jobUpdateError);
    throw new Error(`Failed to update analysis job: ${jobUpdateError.message}`);
  }

  logger.info('[JOB STATUS UPDATED]', {
    jobId,
    fromStatus: currentJob?.result || 'unknown',
    toStatus: results.status,
    updatedAt: data?.updated_at
  });

  return data;
};

export { router as historyRouter };
