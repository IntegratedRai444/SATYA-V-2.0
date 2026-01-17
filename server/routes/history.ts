import { Router, Request, Response, NextFunction } from 'express';
import { body, type ValidationChain } from 'express-validator';
import { validateRequest } from '../middleware/validate-request';
import { authenticate } from '../middleware/auth.middleware';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

const router = Router();

// Extend Express Request type to include user
declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      email: string;
      role: string;
      email_verified: boolean;
      user_metadata?: Record<string, any> | undefined;
    };
  }
}

// Validation chains
const createJobValidation: ValidationChain[] = [
  body('modality').isIn(['image', 'audio', 'video', 'multimodal']).withMessage('Invalid modality'),
  body('status').isIn(['pending', 'processing', 'completed', 'failed']).withMessage('Invalid status'),
  body('filename').notEmpty().withMessage('Filename is required'),
  body('mime_type').notEmpty().withMessage('MIME type is required'),
  body('size_bytes').isInt({ min: 0 }).withMessage('Size must be positive'),
  body('confidence').isFloat({ min: 0, max: 1 }).withMessage('Confidence must be between 0 and 1'),
  body('is_deepfake').isBoolean().withMessage('is_deepfake must be boolean'),
  body('model_name').notEmpty().withMessage('Model name is required'),
  body('model_version').notEmpty().withMessage('Model version is required'),
];

// GET /api/v2/history - Get paginated list of user's analysis jobs
router.get('/',
  authenticate,
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
        .from('analysis_jobs')
        .select(`
          id,
          modality,
          status,
          filename,
          mime_type,
          size_bytes,
          confidence,
          is_deepfake,
          model_name,
          model_version,
          summary,
          created_at,
          completed_at,
          report_code
        `)
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .range(offset, limit);

      if (error) {
        logger.error('Failed to fetch analysis history:', error);
        return res.status(500).json({
          success: false,
          error: 'Failed to fetch analysis history'
        });
      }

      // Get total count for pagination
      const { count } = await supabase
        .from('analysis_jobs')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', userId);

      const totalPages = Math.ceil((count || 0) / limit);

      res.json({
        success: true,
        data: {
          jobs: jobs || [],
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

// GET /api/v2/history/:jobId - Get full job details with analysis results
router.get('/:jobId',
  authenticate,
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

      // Fetch job details
      const { data: job, error: jobError } = await supabase
        .from('analysis_jobs')
        .select(`
          id,
          modality,
          status,
          filename,
          mime_type,
          size_bytes,
          file_hash,
          progress,
          metadata,
          error_message,
          priority,
          retry_count,
          report_code,
          started_at,
          completed_at,
          created_at,
          updated_at,
          confidence,
          is_deepfake,
          model_name,
          model_version,
          summary
        `)
        .eq('id', jobId)
        .eq('user_id', userId)
        .single();

      if (jobError || !job) {
        return res.status(404).json({
          success: false,
          error: 'Analysis job not found'
        });
      }

      // Fetch analysis results for this job
      const { data: results, error: resultsError } = await supabase
        .from('analysis_results')
        .select(`
          id,
          model_name,
          confidence,
          is_deepfake,
          analysis_data,
          proof_json,
          created_at
        `)
        .eq('job_id', jobId)
        .order('created_at', { ascending: false });

      if (resultsError) {
        logger.error('Failed to fetch analysis results:', resultsError);
        // Continue without results rather than failing
      }

      res.json({
        success: true,
        data: {
          job,
          results: results || []
        }
      });
    } catch (error) {
      logger.error('History detail route error:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  }
);

// DELETE /api/v2/history/:jobId - Delete a specific analysis job
router.delete('/:jobId',
  authenticate,
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
        .from('analysis_jobs')
        .select('id')
        .eq('id', jobId)
        .eq('user_id', userId)
        .single();

      if (checkError || !job) {
        return res.status(404).json({
          success: false,
          error: 'Analysis job not found'
        });
      }

      // Delete the job (cascade will delete related analysis_results)
      const { error: deleteError } = await supabase
        .from('analysis_jobs')
        .delete()
        .eq('id', jobId)
        .eq('user_id', userId);

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
    modality: 'image' | 'audio' | 'video' | 'multimodal';
    filename: string;
    mime_type: string;
    size_bytes: number;
    file_hash?: string;
    metadata?: Record<string, any>;
  }
) => {
  const { data, error } = await supabase
    .from('analysis_jobs')
    .insert({
      user_id: userId,
      modality: jobData.modality,
      status: 'processing',
      filename: jobData.filename,
      mime_type: jobData.mime_type,
      size_bytes: jobData.size_bytes,
      file_hash: jobData.file_hash,
      metadata: jobData.metadata || {},
      progress: 0,
      started_at: new Date().toISOString()
    })
    .select('id, report_code')
    .single();

  if (error) {
    logger.error('Failed to create analysis job:', error);
    throw new Error('Failed to create analysis job');
  }

  return data;
};

// Helper function to update analysis job with results
export const updateAnalysisJobWithResults = async (
  jobId: string,
  results: {
    status: 'completed' | 'failed';
    confidence?: number;
    is_deepfake?: boolean;
    model_name?: string;
    model_version?: string;
    summary?: Record<string, any>;
    analysis_data?: Record<string, any>;
    proof_json?: Record<string, any>;
    error_message?: string;
  }
) => {
  // Update the job
  const { data: jobUpdate, error: jobUpdateError } = await supabase
    .from('analysis_jobs')
    .update({
      status: results.status,
      confidence: results.confidence,
      is_deepfake: results.is_deepfake,
      model_name: results.model_name,
      model_version: results.model_version,
      summary: results.summary,
      progress: results.status === 'completed' ? 100 : 0,
      completed_at: results.status === 'completed' ? new Date().toISOString() : null,
      error_message: results.error_message,
      updated_at: new Date().toISOString()
    })
    .eq('id', jobId)
    .select()
    .single();

  if (jobUpdateError) {
    logger.error('Failed to update analysis job:', jobUpdateError);
    throw new Error('Failed to update analysis job');
  }

  // Insert analysis results if provided
  if (results.analysis_data || results.proof_json) {
    const { data: resultInsert, error: resultInsertError } = await supabase
      .from('analysis_results')
      .insert({
        job_id: jobId,
        model_name: results.model_name || 'SatyaAI',
        confidence: results.confidence,
        is_deepfake: results.is_deepfake,
        analysis_data: results.analysis_data || {},
        proof_json: results.proof_json || {}
      })
      .select()
      .single();

    if (resultInsertError) {
      logger.error('Failed to insert analysis results:', resultInsertError);
      throw new Error('Failed to save analysis results');
    }
  }

  return jobUpdate;
};

export default router;
