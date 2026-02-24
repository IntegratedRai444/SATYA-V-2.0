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
          status,
          file_name,
          file_type,
          file_size,
          result,
          report_code,
          created_at,
          completed_at
        `)
        .eq('user_id', userId)
        .eq('type', 'analysis')
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
        .from('tasks')
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
            status: string;
            file_name: string;
            file_type: string;
            file_size: number;
            result?: {
              confidence?: number;
              is_deepfake?: boolean;
              model_name?: string;
              model_version?: string;
              summary?: Record<string, unknown>;
            };
            report_code: string;
            created_at: string;
            completed_at?: string;
          }) => ({
            ...job,
            modality: job.type,
            filename: job.file_name,
            mime_type: job.file_type,
            size_bytes: job.file_size,
            reportCode: job.report_code,  // ADD THIS LINE
            confidence: job.result?.confidence,
            is_deepfake: job.result?.is_deepfake,
            model_name: job.result?.model_name,
            model_version: job.result?.model_version,
            summary: job.result?.summary
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

// GET /api/v2/history/:jobId - Get full job details with analysis results
router.get('/:jobId',
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
        .from('tasks')
        .select(`
          id,
          type,
          status,
          file_name,
          file_type,
          file_size,
          file_path,
          progress,
          metadata,
          error,
          report_code,
          started_at,
          completed_at,
          created_at,
          updated_at
        `)
        .eq('id', jobId)
        .eq('user_id', userId)
        .eq('type', 'analysis')
        .single();

      
      if (jobError || !job) {
        return res.status(404).json({
          success: false,
          error: 'Analysis job not found'
        });
      }

      // Fetch analysis results (stored in tasks.result)
      const jobWithResult = job as {
        result?: {
          confidence?: number;
          is_deepfake?: boolean;
          model_name?: string;
          model_version?: string;
          summary?: Record<string, unknown>;
          analysis_data?: Record<string, unknown>;
          proof_json?: Record<string, unknown>;
          [key: string]: unknown;
        };
      } | null;
      
      const analysisData = jobWithResult?.result;
      const results = analysisData ? [{
        id: job.id,
        model_name: analysisData.model_name || 'SatyaAI',
        confidence: analysisData.confidence,
        is_deepfake: analysisData.is_deepfake,
        analysis_data: analysisData,
        proof_json: analysisData.proof_json || {},
        created_at: job.created_at
      }] : [];

      res.json({
        success: true,
        data: {
          job: {
            ...job,
            modality: job.type,
            filename: job.file_name,
            mime_type: job.file_type,
            size_bytes: job.file_size,
            reportCode: job.report_code,  // ADD THIS LINE
            confidence: analysisData?.confidence,
            is_deepfake: analysisData?.is_deepfake,
            model_name: analysisData?.model_name,
            model_version: analysisData?.model_version,
            summary: analysisData?.summary,
            error_message: job.error
          },
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
        .from('tasks')
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
        .from('tasks')
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
  // Generate report code
  const reportCode = `RPT-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
  
  // Generate jobId if not provided (ensures consistency)
  const finalJobId = jobId || randomUUID();
  
  const { data, error } = await supabaseAdmin
    .from('tasks')
    .insert({
      id: finalJobId, // Use consistent ID
      user_id: userId,
      type: 'analysis',
      status: jobData.status || 'processing',
      file_name: jobData.filename,
      file_type: jobData.mime_type,
      file_size: jobData.size_bytes,
      report_code: reportCode,
      metadata: {
        ...jobData.metadata,
        media_type: jobData.modality
      },
      progress: 0,
      started_at: new Date().toISOString(),
      result: {
        confidence: 0,
        is_deepfake: false,
        model_name: 'SatyaAI',
        model_version: '1.0.0',
        summary: {}
      }
    })
    .select('id, report_code')
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
    returnedId: data?.id,
    reportCode: data?.report_code 
  });

  return data;
};

// 🔥 NEW: Update job with Python worker results
export const updateJobWithResults = async (
  jobId: string,
  result: {
    is_deepfake: boolean;
    confidence: number;
    model_name: string;
    model_version: string;
    analysis_data?: {
      processing_time?: number;
    };
  },
  status: 'completed' | 'failed' = 'completed',
  errorMessage?: string
) => {
  const { data, error: dbError } = await supabaseAdmin
    .from('tasks')
    .update({
      status,
      result: {
        confidence: result.confidence,
        is_deepfake: result.is_deepfake,
        model_name: result.model_name,
        model_version: result.model_version,
        summary: {
          processing_time: result.analysis_data?.processing_time || 0,
          completed_at: new Date().toISOString()
        }
      },
      progress: 100,
      updated_at: new Date().toISOString(),
      error: errorMessage || null
    })
    .eq('id', jobId)
    .select('id, status, updated_at')
    .single();

  if (dbError) {
    logger.error(`Failed to update job ${jobId}:`, dbError);
    const errorMsg = typeof dbError.message === 'string' ? dbError.message : 'Unknown database error';
    throw new Error(`Failed to update job: ${errorMsg}`);
  }

  return data;
};

export const getAnalysisHistory = async (
  userId: string,
  { limit = 20, offset = 0 }: { limit?: number; offset?: number; type?: string } = {}
): Promise<{ items: {
            id: string;
            type: string;
            status: string;
            file_name: string;
            file_type: string;
            file_size: number;
            file_path: string;
            progress: number;
            metadata: Record<string, unknown>;
            error: string | null;
            report_code: string;
            started_at: string;
            completed_at?: string;
            created_at: string;
            updated_at: string;
            deleted_at: string | null;
          }[]; total: number }> => {
  const { data, error } = await supabase
    .from('tasks')
    .select('id, type, status, file_name, file_type, file_size, file_path, progress, metadata, error, report_code, started_at, completed_at, created_at, updated_at, deleted_at')
    .eq('user_id', userId)
    .eq('type', 'analysis')
    .order('created_at', { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) {
    logger.error('Failed to fetch analysis history:', error);
    throw new Error('Failed to fetch analysis history');
  }

  const total = await supabase
    .from('tasks')
    .select('id', { count: 'exact' })
    .eq('user_id', userId)
    .eq('type', 'analysis');

  return { items: data || [], total: total.count || 0 };
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
    summary?: Record<string, unknown>;
    analysis_data?: Record<string, unknown>;
    proof_json?: Record<string, unknown>;
    error_message?: string;
  }
) => {
  // Update the job
  const { data: jobUpdate, error: jobUpdateError } = await supabaseAdmin
    .from('tasks')
    .update({
      status: results.status,
      progress: results.status === 'completed' ? 100 : 0,
      completed_at: results.status === 'completed' ? new Date().toISOString() : null,
      error: results.error_message,
      updated_at: new Date().toISOString(),
      result: {
        confidence: results.confidence || 0,
        is_deepfake: results.is_deepfake || false,
        model_name: results.model_name || 'SatyaAI',
        model_version: results.model_version || '1.0.0',
        summary: results.summary || {},
        analysis_data: results.analysis_data || {},
        proof_json: results.proof_json || {}
      }
    })
    .eq('id', jobId)
    .select()
    .single();

  if (jobUpdateError) {
    logger.error('Failed to update analysis job:', jobUpdateError);
    throw new Error('Failed to update analysis job');
  }

  return jobUpdate;
};

export { router as historyRouter };
