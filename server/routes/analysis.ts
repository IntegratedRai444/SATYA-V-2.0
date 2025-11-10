import { Router, Response } from 'express';
import { requireAuth, AuthenticatedRequest } from '../middleware/auth';
import { uploadMiddleware } from '../middleware/upload';
import { createErrorResponse } from '../types/api-responses';
import axios from 'axios';
import { logger } from '../config/logger';

const router = Router();

// Configuration
const PYTHON_API_BASE = process.env.PYTHON_API_URL || 'http://localhost:5001';

/**
 * Forward file to Python API
 */
const forwardToPython = async (endpoint: string, req: any, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json(createErrorResponse('No file provided'));
    }

    // Forward the file to Python API
    const formData = new FormData();
    formData.append('file', req.file.buffer, { filename: req.file.originalname });

    const response = await axios.post(`${PYTHON_API_BASE}${endpoint}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': req.headers.authorization || ''
      }
    });

    return res.json(response.data);
  } catch (error: any) {
    logger.error(`Error forwarding to Python API: ${error.message}`);
    return res.status(500).json(createErrorResponse('Analysis service unavailable'));
  }
};

/**
 * Forward form data to Python API
 */
const forwardFormDataToPython = async (endpoint: string, req: any, res: Response) => {
  try {
    const formData = new FormData();
    
    // Append files
    if (req.files) {
      for (const file of req.files) {
        formData.append(file.fieldname, file.buffer, { filename: file.originalname });
      }
    }

    // Append other form fields
    for (const [key, value] of Object.entries(req.body)) {
      if (key !== 'file' && key !== 'files') {
        formData.append(key, String(value));
      }
    }

    const response = await axios.post(`${PYTHON_API_BASE}${endpoint}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': req.headers.authorization || ''
      }
    });

    return res.json(response.data);
  } catch (error: any) {
    logger.error(`Error forwarding to Python API: ${error.message}`);
    return res.status(500).json(createErrorResponse('Analysis service unavailable'));
  }
};

// Image Analysis
router.post('/image',
  requireAuth,
  uploadMiddleware.singleImage,
  async (req: AuthenticatedRequest, res: Response) => {
    return forwardToPython('/analyze/image', req, res);
  }
);

// Video Analysis
router.post('/video',
  requireAuth,
  uploadMiddleware.singleVideo,
  async (req: AuthenticatedRequest, res: Response) => {
    return forwardToPython('/analyze/video', req, res);
    }
  }
);

/**
 * POST /api/analysis/audio
 * Analyze uploaded audio for voice cloning detection
 */
router.post('/audio',
  requireAuth,
  cleanupOnError,
  uploadMiddleware.singleAudio,
  handleUploadError,
  validateUploadedFiles,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          message: 'No audio file provided',
          code: 'NO_FILE'
        });
      }

      // Parse analysis options
      const options = analysisOptionsSchema.parse(req.body.options || {});

      // Prepare file info for processing
      const fileInfo = {
        audio: {
          originalName: req.file.originalname,
          filename: req.file.filename,
          path: req.file.path,
          size: req.file.size,
          mimeType: req.file.mimetype
        }
      };

      if (options.async) {
        // Process asynchronously
        const jobId = await fileProcessor.addJob(
          req.user!.userId,
          'audio',
          fileInfo,
          options
        );

        logger.info('Async audio analysis started', {
          jobId,
          userId: req.user!.userId,
          filename: req.file.filename
        });

        res.json({
          success: true,
          message: 'Audio analysis started',
          jobId,
          async: true,
          estimatedTime: 60 // seconds
        });
      } else {
        // Process synchronously
        const audioBuffer = require('fs').readFileSync(req.file.path);
        const token = req.headers.authorization?.replace('Bearer ', '') || null;

        const result = await pythonBridge.analyzeAudio(audioBuffer, req.file.originalname, token);

        // Clean up uploaded file
        require('fs').unlinkSync(req.file.path);

        logger.info('Sync audio analysis completed', {
          userId: req.user!.userId,
          filename: req.file.filename,
          result: result.success ? 'success' : 'failed'
        });

        res.json({
          success: true,
          message: 'Audio analysis completed',
          result: {
            ...result,
            fileInfo: {
              originalName: req.file.originalname,
              size: req.file.size,
              mimeType: req.file.mimetype
            }
          },
          async: false
        });
      }

    } catch (error) {
      logger.error('Audio analysis error', {
        error: (error as Error).message,
        userId: req.user?.userId,
        filename: req.file?.filename
      });

      res.status(500).json({
        success: false,
        message: 'Audio analysis failed',
        code: 'ANALYSIS_ERROR',
        error: (error as Error).message
      });
    }
  }
);

/**
 * POST /api/analysis/multimodal
 * Analyze multiple files for comprehensive deepfake detection
 */
router.post('/multimodal',
  requireAuth,
  cleanupOnError,
  uploadMiddleware.multimodal,
  handleUploadError,
  validateUploadedFiles,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      
      if (!files || Object.keys(files).length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files provided for multimodal analysis',
          code: 'NO_FILES'
        });
      }

      // Parse analysis options
      const options = analysisOptionsSchema.parse(req.body.options || {});

      // Prepare file info for processing
      const fileInfo: any = {};
      let totalSize = 0;

      for (const [fieldName, fileArray] of Object.entries(files)) {
        if (fileArray && fileArray.length > 0) {
          const file = fileArray[0];
          fileInfo[fieldName] = {
            originalName: file.originalname,
            filename: file.filename,
            path: file.path,
            size: file.size,
            mimeType: file.mimetype
          };
          totalSize += file.size;
        }
      }

      if (options.async) {
        // Process asynchronously (recommended for multimodal)
        const jobId = await fileProcessor.addJob(
          req.user!.userId,
          'multimodal',
          fileInfo,
          options
        );

        logger.info('Async multimodal analysis started', {
          jobId,
          userId: req.user!.userId,
          fileTypes: Object.keys(fileInfo),
          totalSize
        });

        res.json({
          success: true,
          message: 'Multimodal analysis started',
          jobId,
          async: true,
          fileTypes: Object.keys(fileInfo),
          estimatedTime: Math.max(180, Math.round(totalSize / (1024 * 1024) * 2)) // Estimate based on total size
        });
      } else {
        // Process synchronously (not recommended for multimodal)
        const imageBuffer = fileInfo.image ? require('fs').readFileSync(fileInfo.image.path) : null;
        const videoBuffer = fileInfo.video ? require('fs').readFileSync(fileInfo.video.path) : null;
        const audioBuffer = fileInfo.audio ? require('fs').readFileSync(fileInfo.audio.path) : null;

        const token = req.headers.authorization?.replace('Bearer ', '') || null;

        // Use the first available buffer for analysis
        const primaryBuffer = imageBuffer || videoBuffer || audioBuffer;
        const analysisType = imageBuffer ? 'image' : videoBuffer ? 'video' : 'audio';
        
        let result;
        if (analysisType === 'image') {
          result = await pythonBridge.analyzeImage(primaryBuffer, token);
        } else if (analysisType === 'video') {
          result = await pythonBridge.analyzeVideo(primaryBuffer, 'multimodal.mp4', token);
        } else {
          result = await pythonBridge.analyzeAudio(primaryBuffer, 'multimodal.wav', token);
        }

        // Clean up uploaded files
        for (const file of Object.values(fileInfo)) {
          require('fs').unlinkSync((file as any).path);
        }

        logger.info('Sync multimodal analysis completed', {
          userId: req.user!.userId,
          fileTypes: Object.keys(fileInfo),
          result: result.success ? 'success' : 'failed'
        });

        res.json({
          success: true,
          message: 'Multimodal analysis completed',
          result: {
            ...result,
            fileInfo: Object.fromEntries(
              Object.entries(fileInfo).map(([key, file]) => [
                key,
                {
                  originalName: (file as any).originalName,
                  size: (file as any).size,
                  mimeType: (file as any).mimeType
                }
              ])
            )
          },
          async: false
        });
      }

    } catch (error) {
      logger.error('Multimodal analysis error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Multimodal analysis failed',
        code: 'ANALYSIS_ERROR',
        error: (error as Error).message
      });
    }
  }
);

/**
 * POST /api/analysis/webcam
 * Analyze webcam image data for real-time detection
 */
router.post('/webcam',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      // Validate request body
      const validatedData = webcamAnalysisSchema.parse(req.body);
      
      // Convert base64 image data to buffer
      const base64Data = validatedData.imageData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({
          success: false,
          message: 'Invalid image data format',
          code: 'INVALID_IMAGE_DATA'
        });
      }

      const imageBuffer = Buffer.from(base64Data, 'base64');
      const token = req.headers.authorization?.replace('Bearer ', '') || null;

      // Process webcam images synchronously for real-time feedback
      const result = await pythonBridge.analyzeImage(imageBuffer, token);

      logger.info('Webcam analysis completed', {
        userId: req.user!.userId,
        imageSize: imageBuffer.length,
        result: result.success ? 'success' : 'failed'
      });

      res.json({
        success: true,
        message: 'Webcam analysis completed',
        result: {
          ...result,
          source: 'webcam',
          timestamp: new Date().toISOString()
        }
      });

    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          message: 'Invalid request data',
          errors: error.errors.map(err => ({
            field: err.path.join('.'),
            message: err.message
          }))
        });
      }

      logger.error('Webcam analysis error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Webcam analysis failed',
        code: 'ANALYSIS_ERROR',
        error: (error as Error).message
      });
    }
  }
);

/**
 * GET /api/analysis/:id/status
 * Get real-time analysis status
 */
router.get('/:id/status',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { id } = req.params;
      
      if (!id) {
        return res.status(400).json({
          success: false,
          message: 'Analysis ID is required',
          code: 'MISSING_ID'
        });
      }

      const job = fileProcessor.getJob(id);
      
      if (!job) {
        return res.status(404).json({
          success: false,
          message: 'Analysis not found',
          code: 'NOT_FOUND'
        });
      }

      // Check if user has permission to view this job
      if (req.user?.userId !== job.userId) {
        return res.status(403).json({
          success: false,
          message: 'Access denied',
          code: 'ACCESS_DENIED'
        });
      }

      res.json({
        success: true,
        data: {
          analysisId: job.id,
          status: job.status,
          progress: {
            percentage: job.progress.percentage || 0,
            stage: job.progress.stage || 'initializing',
            estimatedTimeRemaining: job.progress.estimatedTimeRemaining || null
          },
          type: job.type,
          createdAt: job.createdAt,
          startedAt: job.progress.startTime,
          completedAt: job.progress.endTime,
          processingTime: job.progress.endTime && job.progress.startTime ? 
            job.progress.endTime.getTime() - job.progress.startTime.getTime() : null,
          error: job.error || null
        }
      });

    } catch (error) {
      logger.error('Failed to get analysis status', {
        error: (error as Error).message,
        analysisId: req.params.id,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve analysis status',
        code: 'STATUS_ERROR',
        error: (error as Error).message
      });
    }
  }
);

/**
 * GET /api/analysis/:id/results
 * Get completed analysis results
 */
router.get('/:id/results',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { id } = req.params;
      
      if (!id) {
        return res.status(400).json({
          success: false,
          message: 'Analysis ID is required',
          code: 'MISSING_ID'
        });
      }

      const job = fileProcessor.getJob(id);
      
      if (!job) {
        return res.status(404).json({
          success: false,
          message: 'Analysis not found',
          code: 'NOT_FOUND'
        });
      }

      // Check if user has permission to view this job
      if (req.user?.userId !== job.userId) {
        return res.status(403).json({
          success: false,
          message: 'Access denied',
          code: 'ACCESS_DENIED'
        });
      }

      if (job.status !== 'completed') {
        return res.status(400).json({
          success: false,
          message: `Analysis is not completed. Current status: ${job.status}`,
          code: 'NOT_COMPLETED',
          data: {
            status: job.status,
            progress: job.progress
          }
        });
      }

      if (!job.result) {
        return res.status(404).json({
          success: false,
          message: 'Analysis results not found',
          code: 'NO_RESULTS'
        });
      }

      res.json({
        success: true,
        data: {
          analysisId: job.id,
          results: job.result,
          confidence: job.result.confidence || 0,
          processingTime: job.progress.endTime && job.progress.startTime ? 
            job.progress.endTime.getTime() - job.progress.startTime.getTime() : null,
          metadata: {
            type: job.type,
            createdAt: job.createdAt,
            completedAt: job.progress.endTime,
            fileInfo: job.files
          }
        }
      });

    } catch (error) {
      logger.error('Failed to get analysis results', {
        error: (error as Error).message,
        analysisId: req.params.id,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve analysis results',
        code: 'RESULTS_ERROR',
        error: (error as Error).message
      });
    }
  }
);

/**
 * GET /api/analysis/result/:jobId
 * Get analysis result for a completed job (legacy endpoint)
 */
router.get('/result/:jobId',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { jobId } = req.params;
      
      if (!jobId) {
        return res.status(400).json({
          success: false,
          message: 'Job ID is required'
        });
      }

      const job = fileProcessor.getJob(jobId);
      
      if (!job) {
        return res.status(404).json({
          success: false,
          message: 'Job not found'
        });
      }

      // Check if user has permission to view this job
      if (req.user?.userId !== job.userId) {
        return res.status(403).json({
          success: false,
          message: 'Access denied'
        });
      }

      if (job.status !== 'completed') {
        return res.status(400).json({
          success: false,
          message: `Job is not completed. Current status: ${job.status}`,
          status: job.status,
          progress: job.progress
        });
      }

      res.json({
        success: true,
        message: 'Analysis result retrieved',
        result: job.result,
        jobInfo: {
          id: job.id,
          type: job.type,
          status: job.status,
          createdAt: job.createdAt,
          completedAt: job.progress.endTime,
          processingTime: job.progress.endTime ? 
            job.progress.endTime.getTime() - job.progress.startTime.getTime() : null
        }
      });

    } catch (error) {
      logger.error('Failed to get analysis result', {
        error: (error as Error).message,
        jobId: req.params.jobId,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve analysis result'
      });
    }
  }
);

/**
 * GET /api/analysis/history
 * Get user's analysis history
 */
router.get('/history',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      // Parse query parameters
      const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);
      const offset = parseInt(req.query.offset as string) || 0;
      const type = req.query.type as string;
      const status = req.query.status as string;

      let jobs = fileProcessor.getUserJobs(req.user.userId);

      // Filter by type if specified
      if (type && ['image', 'video', 'audio', 'multimodal'].includes(type)) {
        jobs = jobs.filter(job => job.type === type);
      }

      // Filter by status if specified
      if (status && ['completed', 'failed', 'processing', 'queued'].includes(status)) {
        jobs = jobs.filter(job => job.status === status);
      }

      // Sort by creation date (newest first)
      jobs.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

      // Apply pagination
      const paginatedJobs = jobs.slice(offset, offset + limit);

      // Format response
      const history = paginatedJobs.map(job => ({
        id: job.id,
        type: job.type,
        status: job.status,
        createdAt: job.createdAt,
        completedAt: job.progress.endTime,
        processingTime: job.progress.endTime ? 
          job.progress.endTime.getTime() - job.progress.startTime.getTime() : null,
        fileCount: Object.keys(job.files).length,
        hasResult: !!job.result,
        error: job.error
      }));

      res.json({
        success: true,
        history,
        pagination: {
          total: jobs.length,
          limit,
          offset,
          hasMore: offset + limit < jobs.length
        }
      });

    } catch (error) {
      logger.error('Failed to get analysis history', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve analysis history'
      });
    }
  }
);

export default router;