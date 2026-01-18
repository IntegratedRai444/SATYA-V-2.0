import { Router, Request, Response, NextFunction, RequestHandler } from 'express';
import { pythonBridge } from '../services/python-bridge';
import { supabaseAuth } from '../middleware/supabase-auth';
import { v4 as uuidv4 } from 'uuid';
import multer from 'multer';
import { SupabaseUser } from '../types/supabase';
import { createAnalysisJob, updateAnalysisJobWithResults } from './history';
import { supabase } from '../config/supabase';

// Type for authenticated requests
type AuthenticatedRequest = Request & {
  user: {
    id: string;
    email: string;
    role: string;
    user_metadata?: Record<string, any>;
  };
};
import path from 'path';
import fs from 'fs';
import { promisify } from 'util';
import { logger } from '../config/logger';

const router = Router();
const upload = multer({ dest: 'uploads/' });
const unlinkAsync = promisify(fs.unlink);

// Helper function to determine file type
const getFileType = (mimetype: string): 'image' | 'video' | 'audio' => {
  if (mimetype.startsWith('image/')) return 'image';
  if (mimetype.startsWith('video/')) return 'video';
  if (mimetype.startsWith('audio/')) return 'audio';
  throw new Error('Unsupported file type');
};

// Submit file for analysis
router.post(
  '/analyze',
  supabaseAuth as RequestHandler,
  upload.single('file'),
  async (req: Request, res: Response, next: NextFunction) => {
    const authReq = req as unknown as AuthenticatedRequest;
    
    if (!authReq.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const fileType = getFileType(req.file.mimetype);
      
      // Create analysis job in database first
      const jobData = await createAnalysisJob(authReq.user.id, {
        modality: fileType,
        filename: req.file.originalname,
        mime_type: req.file.mimetype,
        size_bytes: req.file.size,
        metadata: {
          originalName: req.file.originalname,
          mimeType: req.file.mimetype,
          size: req.file.size,
        },
      });
      
      const analysisId = jobData.id;
      
      // In a production environment, you might want to move the file to persistent storage
      // For now, we'll use the temporary upload location
      const filePath = req.file.path;

      try {
        const analysisResult = await pythonBridge.analyzeMedia({
          filePath,
          fileType,
          userId: authReq.user.id,
          metadata: {
            originalName: req.file.originalname,
            mimeType: req.file.mimetype,
            size: req.file.size,
          },
        });

        // Save analysis results to database
        await updateAnalysisJobWithResults(jobData.id, {
          status: 'completed',
          confidence: analysisResult.result?.confidence || 0,
          is_deepfake: analysisResult.result?.isDeepfake || false,
          model_name: analysisResult.result?.modelUsed || 'SatyaAI',
          model_version: '1.0.0', // Default version since not in response
          summary: analysisResult.result?.analysisDetails || {},
          analysis_data: analysisResult.result || {},
          proof_json: analysisResult.result ? {
            model_name: analysisResult.result.modelUsed || 'SatyaAI',
            model_version: '1.0.0',
            processing_time: analysisResult.result.processingTime || 0,
            confidence: analysisResult.result.confidence || 0,
            timestamp: new Date().toISOString()
          } : {}
        });

        // Clean up the uploaded file
        await unlinkAsync(filePath).catch(err => {
          logger.warn('Failed to delete temporary file', { filePath, error: err.message });
        });

        return res.json({
          id: analysisId,
          status: 'completed',
          ...analysisResult,
        });
      } catch (error) {
        // Update job with error status
        try {
          await updateAnalysisJobWithResults(jobData.id, {
            status: 'failed',
            error_message: error instanceof Error ? error.message : 'Unknown error'
          });
        } catch (dbError) {
          logger.error('Failed to update job with error status:', dbError);
        }
        
        // Ensure we clean up the file even if analysis fails
        await unlinkAsync(filePath).catch(err => {
          logger.warn('Failed to delete temporary file after error', { 
            filePath, 
            error: err.message 
          });
        });
        throw error;
      }
    } catch (error) {
      next(error);
    }
  }
);

// Get analysis status
router.get('/analysis/status/:id', supabaseAuth, async (req, res, next) => {
  try {
    const { id } = req.params;
    const status = await pythonBridge.getAnalysisStatus(id);
    res.json(status);
  } catch (error) {
    next(error);
  }
});

export default router;
