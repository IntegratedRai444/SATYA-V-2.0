import { Router, Request, Response } from 'express';
import multer, { FileFilterCallback } from 'multer';

// Interface for multer file in fileFilter
interface MulterFile {
  fieldname: string;
  originalname: string;
  encoding: string;
  mimetype: string;
  size: number;
  destination: string;
  filename: string;
  path: string;
  buffer: Buffer;
}
import path from 'path';
import { validateRequest } from '../middleware/validate-request';
import { logger } from '../config/logger';
import { pythonBridge } from '../services/python-http-bridge';
import { createAnalysisJob, updateAnalysisJobWithResults } from './history';
import FormData from 'form-data';
import { randomUUID } from 'crypto';
import { AuthenticatedRequest } from '../types/auth';
import webSocketManager from '../services/websocket-manager';
import { setImmediate } from 'timers';

type AnalysisModality = 'image' | 'video' | 'audio' | 'text';

// Helper function to send job progress updates
const sendJobProgress = (userId: string, jobId: string, stage: string, progress: number) => {
  webSocketManager.sendEventToUser(userId, {
    type: 'JOB_PROGRESS',
    jobId,
    timestamp: Date.now(),
    data: {
      stage,
      progress
    }
  });
};

const router = Router();

// Configure multer for secure file uploads with memory storage
const storage = multer.memoryStorage();

// Define allowed file types with their magic numbers and size limits (in bytes)
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_FILE_TYPES = {
  'image/jpeg': { 
    ext: '.jpg', 
    magic: [0xFF, 0xD8, 0xFF],
    maxSize: 10 * 1024 * 1024 // 10MB for JPEG
  },
  'image/jpg': { 
    ext: '.jpg', 
    magic: [0xFF, 0xD8, 0xFF],
    maxSize: 10 * 1024 * 1024 // 10MB for JPEG
  },
  'image/png': { 
    ext: '.png', 
    magic: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
    maxSize: 10 * 1024 * 1024 // 10MB for PNG
  },
  'image/webp': { 
    ext: '.webp', 
    magic: [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0x57, 0x45, 0x42, 0x50],
    maxSize: 10 * 1024 * 1024 // 10MB for WebP
  },
  'video/mp4': { 
    ext: '.mp4', 
    magic: [0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70],
    maxSize: MAX_FILE_SIZE
  },
  'video/webm': { 
    ext: '.webm', 
    magic: [0x1A, 0x45, 0xDF, 0xA3],
    maxSize: MAX_FILE_SIZE
  },
  'video/avi': { 
    ext: '.avi', 
    magic: [0x52, 0x49, 0x46, 0x46],
    maxSize: MAX_FILE_SIZE
  },
  'video/mov': { 
    ext: '.mov', 
    magic: [0x6D, 0x6F, 0x6F, 0x76],
    maxSize: MAX_FILE_SIZE
  },
  'video/mkv': { 
    ext: '.mkv', 
    magic: [0x1A, 0x45, 0xDF, 0xA3],
    maxSize: MAX_FILE_SIZE
  },
  'audio/mp3': { 
    ext: '.mp3', 
    magic: [0x49, 0x44, 0x33],
    maxSize: MAX_FILE_SIZE
  },
  'audio/wav': { 
    ext: '.wav', 
    magic: [0x52, 0x49, 0x46, 0x46],
    maxSize: MAX_FILE_SIZE
  },
  'audio/mpeg': { 
    ext: '.mp3', 
    magic: [0x49, 0x44, 0x33],
    maxSize: MAX_FILE_SIZE
  },
  'audio/ogg': { 
    ext: '.ogg', 
    magic: [0x4F, 0x67, 0x67, 0x53],
    maxSize: MAX_FILE_SIZE
  },
} as const;

type AllowedMimeType = keyof typeof ALLOWED_FILE_TYPES;

// Function to check file magic numbers
const checkMagicNumbers = (buffer: Buffer, expectedMagic: readonly number[]): boolean => {
  if (buffer.length < expectedMagic.length) return false;
  
  for (let i = 0; i < expectedMagic.length; i++) {
    if (buffer[i] !== expectedMagic[i]) {
      return false;
    }
  }
  return true;
};

const upload = multer({
  storage,
  limits: {
    fileSize: MAX_FILE_SIZE,
    files: 1,
    fields: 5, // Limit number of form fields
    headerPairs: 20, // Limit number of header key-value pairs
  },
  fileFilter: (req: Request, file: MulterFile, cb: FileFilterCallback) => {
    try {
      // Validate MIME type
      const mimeType = file.mimetype as AllowedMimeType;
      const fileConfig = ALLOWED_FILE_TYPES[mimeType];
      
      if (!fileConfig) {
        return cb(new Error(`Unsupported file type: ${file.mimetype}. Allowed types: ${Object.keys(ALLOWED_FILE_TYPES).join(', ')}`));
      }

      // Validate file extension
      const ext = path.extname(file.originalname).toLowerCase();
      if (ext !== fileConfig.ext) {
        return cb(new Error(`Invalid file extension. Expected ${fileConfig.ext} for ${file.mimetype}`));
      }

      // Validate filename (allow common valid characters)
      const sanitizedFilename = path.basename(file.originalname).replace(/[<>:"/\\|?*\x00-\x1F]/g, '');
      if (sanitizedFilename !== file.originalname) {
        return cb(new Error('Invalid characters in filename'));
      }

      // Check for path traversal
      if (file.originalname.includes('..') || path.isAbsolute(file.originalname)) {
        return cb(new Error('Invalid file path'));
      }

      // Check file size against type-specific limit
      if (file.size > fileConfig.maxSize) {
        return cb(new Error(`File too large. Maximum size: ${formatBytes(fileConfig.maxSize)}`));
      }

      // Additional security checks
      cb(null, true);

      // For small files, we can check magic numbers immediately
      if (file.size > 0 && file.size < fileConfig.magic.length) {
        return cb(new Error('File too small to determine type'));
      }

      // Check magic numbers for files that have enough data
      if (file.size >= fileConfig.magic.length) {
        const magic = file.buffer.slice(0, fileConfig.magic.length);
        if (!checkMagicNumbers(magic, fileConfig.magic)) {
          return cb(new Error('Invalid file signature'));
        }
      }

      cb(null, true);
    } catch (error: unknown) {
      const err = error instanceof Error ? error : new Error('Unknown error during file validation');
      logger.error('File validation error', { 
        error: err.message, 
        filename: file.originalname,
        mimetype: file.mimetype,
        size: file.size
      });
      cb(err);
    }
  },
});

// Helper function to format bytes
function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Helper function to handle analysis errors with proper typing and security
const handleAnalysisError = async (error: unknown, res: Response, jobId?: string, context: Record<string, unknown> = {}) => {
  // Create a safe error object
  const safeError = error instanceof Error 
    ? { 
        name: error.name,
        message: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      }
    : { message: 'An unknown error occurred' };
  
  // Extract Python bridge error details if available
  const pythonError = error && typeof error === 'object' && 'details' in error ? error.details as {
    code?: string;
    message?: string;
    status?: number;
    timestamp?: string;
  } : undefined;
  
  // Determine appropriate status code and message
  let statusCode = 500;
  let errorCode = 'ANALYSIS_ERROR';
  let message = 'Analysis failed. Please try again.';
  
  if (pythonError?.code) {
    switch (pythonError.code) {
      case 'PYTHON_DOWN':
        statusCode = 503;
        errorCode = 'AI_ENGINE_UNAVAILABLE';
        message = 'AI engine temporarily unavailable. Please try again later.';
        break;
      case 'ANALYSIS_TIMEOUT':
        statusCode = 504;
        errorCode = 'ANALYSIS_TIMEOUT';
        message = 'Analysis took too long. Please try with a smaller file.';
        break;
      case 'INVALID_REQUEST':
      case 'VALIDATION_ERROR':
        statusCode = 400;
        errorCode = 'INVALID_FILE';
        message = 'Invalid file format or corrupted file.';
        break;
      case 'NETWORK_ERROR':
        statusCode = 503;
        errorCode = 'NETWORK_ERROR';
        message = 'Network connection to AI service failed.';
        break;
      default:
        statusCode = 502;
        errorCode = 'AI_SERVICE_ERROR';
        message = 'AI service error occurred.';
    }
  }
  
  // If we have a jobId, update it with failed status
  if (jobId) {
    try {
      await updateAnalysisJobWithResults(jobId, {
        status: 'failed',
        error_message: pythonError?.message || safeError.message
      });
    } catch (dbError) {
      logger.error('Failed to update job status:', dbError);
    }
  }
  
  // Log the error with context (redacting sensitive data)
  logger.error('Analysis error', {
    ...safeError,
    ...context,
    pythonError,
    jobId,
    // Redact sensitive data from context
    headers: undefined,
    body: undefined,
    file: context.file ? '[REDACTED]' : undefined,
  });

  // Return a sanitized error response
  return res.status(statusCode).json({
    success: false,
    code: errorCode,
    message,
    details: pythonError || undefined,
    // Only include error details in development
    ...(process.env.NODE_ENV === 'development' && { 
      error: safeError.message,
      pythonError 
    }),
    // Include a request ID for support
    requestId: res.locals.requestId,
    ...(jobId && { jobId })
  });
};

// Image analysis endpoint
router.post(
  '/image',
  upload.single('file'),
  validateRequest([]),
  async (req: AuthenticatedRequest, res: Response) => {
    const type = 'image';
    const correlationId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const file = req.file;
    const userId = req.user?.id;

    logger.info(`[ROUTE] Incoming ${type} analysis request`, {
      correlationId,
      type,
      userId,
      filename: file?.originalname,
      fileSize: file?.size,
    });

    try {
      // Validate authentication
      if (!userId) {
        logger.warn(`[AUTH] User not authenticated for ${type} analysis`, { correlationId });
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      // Validate file
      if (!file) {
        logger.warn(`[IMAGE] No file uploaded`, { correlationId });
        return res.status(400).json({
          success: false,
          message: 'No file uploaded',
        });
      }

      logger.info(`[IMAGE] File parsed successfully`, {
        correlationId,
        filename: file.originalname,
        mimetype: file.mimetype,
        size: file.size
      });

      // Create job with standardized ID
      const jobId = `job_${randomUUID()}`;
      
      // Create job in database
      const job = await createAnalysisJob(userId || '', {
        modality: type as AnalysisModality,
        filename: file.originalname,
        mime_type: file.mimetype,
        size_bytes: file.size,
        metadata: {
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
        },
      });

      logger.info(`[ANALYSIS JOB] Created job ${jobId}`, {
        correlationId,
        jobId,
        type,
      });

      // Send immediate response with job_id
      res.status(202).json({
        success: true,
        job_id: jobId,
        status: 'processing',
      });

      // Process in background
      setImmediate(async () => {
        try {
          // Send progress update
          if (userId) {
            sendJobProgress(userId, jobId, 'uploading', 10);
          }

          // Forward to Python service for inference only
          const formData = new FormData();
          formData.append('file', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype,
          });
          formData.append('job_id', jobId);
          formData.append('user_id', userId || '');

          if (userId) {
            sendJobProgress(userId, jobId, 'analyzing', 30);
          }

          const pythonResponse = await pythonBridge.post(`/api/v2/analysis/unified/${type}`, formData, {
            timeout: 300000, // 5 minutes
          }) as {
            data: {
              confidence: number;
              is_deepfake: boolean;
              model_name: string;
              model_version: string;
              analysis_data: Record<string, unknown>;
              proof: Record<string, unknown>;
            };
          };

          if (userId) {
            sendJobProgress(userId, jobId, 'processing', 70);
          }

          // Python returns inference-only payload
          const inferenceResult = pythonResponse.data;
          
          // Node owns the job lifecycle - update with results
          await updateAnalysisJobWithResults(job.id, {
            status: 'completed',
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
            model_name: inferenceResult.model_name,
            model_version: inferenceResult.model_version,
            analysis_data: inferenceResult.analysis_data,
            proof_json: inferenceResult.proof,
          });

          if (userId) {
            sendJobProgress(userId, jobId, 'completed', 100);

            // Send WebSocket event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_COMPLETED',
              jobId,
              timestamp: Date.now(),
              data: {
                status: 'completed',
                confidence: inferenceResult.confidence,
                is_deepfake: inferenceResult.is_deepfake,
                model_name: inferenceResult.model_name,
                model_version: inferenceResult.model_version,
              },
            });
          }

          logger.info(`[ANALYSIS COMPLETED] Job ${jobId} finished successfully`, {
            correlationId,
            jobId,
            type,
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
          });

        } catch (error) {
          // Handle analysis error
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? ((error as unknown) as { response: { data?: { detail?: string } } }).response?.data?.detail || ((error as unknown) as Error).message || 'Analysis failed'
            : ((error as unknown) as Error).message || 'Analysis failed';
          
          await updateAnalysisJobWithResults(job.id, {
            status: 'failed',
            error_message: errorMessage,
          });

          if (userId) {
            // Send WebSocket error event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_FAILED',
              jobId,
              timestamp: Date.now(),
              data: {
                error: errorMessage,
              },
            });
          }

          logger.error(`[ANALYSIS FAILED] Job ${jobId} failed`, {
            correlationId,
            jobId,
            type,
            error: errorMessage,
          });
        }
      });

    } catch (error) {
      logger.error(`[ANALYSIS ERROR] ${type} analysis failed`, {
        correlationId,
        type,
        error: (error as Error).message,
      });
      
      return res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }
);

// Video analysis endpoint
router.post(
  '/video',
  upload.single('file'),
  validateRequest([]),
  async (req: AuthenticatedRequest, res: Response) => {
    const type = 'video';
    const correlationId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const file = req.file;
    const userId = req.user?.id;

    logger.info(`[ANALYSIS REQUEST] ${type} analysis started`, {
      correlationId,
      type,
      userId,
      filename: file?.originalname,
      fileSize: file?.size,
    });

    try {
      // Validate file
      if (!file) {
        return res.status(400).json({
          success: false,
          message: 'No file uploaded',
        });
      }

      // Create job with standardized ID
      const jobId = `job_${randomUUID()}`;
      
      // Create job in database
      const job = await createAnalysisJob(userId || '', {
        modality: type as AnalysisModality,
        filename: file.originalname,
        mime_type: file.mimetype,
        size_bytes: file.size,
        metadata: {
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
        },
      });

      logger.info(`[ANALYSIS JOB] Created job ${jobId}`, {
        correlationId,
        jobId,
        type,
      });

      // Send immediate response with job_id
      res.status(202).json({
        success: true,
        job_id: jobId,
        status: 'processing',
      });

      // Process in background
      setImmediate(async () => {
        try {
          // Send progress update
          if (userId) {
            sendJobProgress(userId, jobId, 'uploading', 10);
          }

          // Forward to Python service for inference only
          const formData = new FormData();
          formData.append('file', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype,
          });
          formData.append('job_id', jobId);
          formData.append('user_id', userId || '');

          if (userId) {
            sendJobProgress(userId, jobId, 'analyzing', 30);
          }

          const pythonResponse = await pythonBridge.post(`/api/v2/analysis/unified/${type}`, formData, {
            timeout: 300000, // 5 minutes
          }) as {
            data: {
              confidence: number;
              is_deepfake: boolean;
              model_name: string;
              model_version: string;
              analysis_data: Record<string, unknown>;
              proof: Record<string, unknown>;
            };
          };

          if (userId) {
            sendJobProgress(userId, jobId, 'processing', 70);
          }

          // Python returns inference-only payload
          const inferenceResult = pythonResponse.data;
          
          // Node owns the job lifecycle - update with results
          await updateAnalysisJobWithResults(job.id, {
            status: 'completed',
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
            model_name: inferenceResult.model_name,
            model_version: inferenceResult.model_version,
            analysis_data: inferenceResult.analysis_data,
            proof_json: inferenceResult.proof,
          });

          if (userId) {
            sendJobProgress(userId, jobId, 'completed', 100);

            // Send WebSocket event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_COMPLETED',
              jobId,
              timestamp: Date.now(),
              data: {
                status: 'completed',
                confidence: inferenceResult.confidence,
                is_deepfake: inferenceResult.is_deepfake,
                model_name: inferenceResult.model_name,
                model_version: inferenceResult.model_version,
              },
            });
          }

          logger.info(`[ANALYSIS COMPLETED] Job ${jobId} finished successfully`, {
            correlationId,
            jobId,
            type,
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
          });

        } catch (error) {
          // Handle analysis error
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? ((error as unknown) as { response: { data?: { detail?: string } } }).response?.data?.detail || ((error as unknown) as Error).message || 'Analysis failed'
            : ((error as unknown) as Error).message || 'Analysis failed';
          
          await updateAnalysisJobWithResults(job.id, {
            status: 'failed',
            error_message: errorMessage,
          });

          if (userId) {
            // Send WebSocket error event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_FAILED',
              jobId,
              timestamp: Date.now(),
              data: {
                error: errorMessage,
              },
            });
          }

          logger.error(`[ANALYSIS FAILED] Job ${jobId} failed`, {
            correlationId,
            jobId,
            type,
            error: errorMessage,
          });
        }
      });

    } catch (error) {
      logger.error(`[ANALYSIS ERROR] ${type} analysis failed`, {
        correlationId,
        type,
        error: (error as Error).message,
      });
      
      return res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }
);

// Audio analysis endpoint
router.post(
  '/audio',
  upload.single('file'),
  validateRequest([]),
  async (req: AuthenticatedRequest, res: Response) => {
    const type = 'audio';
    const correlationId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const file = req.file;
    const userId = req.user?.id;

    logger.info(`[ANALYSIS REQUEST] ${type} analysis started`, {
      correlationId,
      type,
      userId,
      filename: file?.originalname,
      fileSize: file?.size,
    });

    try {
      // Validate file
      if (!file) {
        return res.status(400).json({
          success: false,
          message: 'No file uploaded',
        });
      }

      // Create job with standardized ID
      const jobId = `job_${randomUUID()}`;
      
      // Create job in database
      const job = await createAnalysisJob(userId || '', {
        modality: type as AnalysisModality,
        filename: file.originalname,
        mime_type: file.mimetype,
        size_bytes: file.size,
        metadata: {
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
        },
      });

      logger.info(`[ANALYSIS JOB] Created job ${jobId}`, {
        correlationId,
        jobId,
        type,
      });

      // Send immediate response with job_id
      res.status(202).json({
        success: true,
        job_id: jobId,
        status: 'processing',
      });

      // Process in background
      setImmediate(async () => {
        try {
          // Send progress update
          if (userId) {
            sendJobProgress(userId, jobId, 'uploading', 10);
          }

          // Forward to Python service for inference only
          const formData = new FormData();
          formData.append('file', file.buffer, {
            filename: file.originalname,
            contentType: file.mimetype,
          });
          formData.append('job_id', jobId);
          formData.append('user_id', userId || '');

          if (userId) {
            sendJobProgress(userId, jobId, 'analyzing', 30);
          }

          const pythonResponse = await pythonBridge.post(`/api/v2/analysis/unified/${type}`, formData, {
            timeout: 300000, // 5 minutes
          }) as {
            data: {
              confidence: number;
              is_deepfake: boolean;
              model_name: string;
              model_version: string;
              analysis_data: Record<string, unknown>;
              proof: Record<string, unknown>;
            };
          };

          if (userId) {
            sendJobProgress(userId, jobId, 'processing', 70);
          }

          // Python returns inference-only payload
          const inferenceResult = pythonResponse.data;
          
          // Node owns the job lifecycle - update with results
          await updateAnalysisJobWithResults(job.id, {
            status: 'completed',
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
            model_name: inferenceResult.model_name,
            model_version: inferenceResult.model_version,
            analysis_data: inferenceResult.analysis_data,
            proof_json: inferenceResult.proof,
          });

          if (userId) {
            sendJobProgress(userId, jobId, 'completed', 100);

            // Send WebSocket event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_COMPLETED',
              jobId,
              timestamp: Date.now(),
              data: {
                status: 'completed',
                confidence: inferenceResult.confidence,
                is_deepfake: inferenceResult.is_deepfake,
                model_name: inferenceResult.model_name,
                model_version: inferenceResult.model_version,
              },
            });
          }

          logger.info(`[ANALYSIS COMPLETED] Job ${jobId} finished successfully`, {
            correlationId,
            jobId,
            type,
            confidence: inferenceResult.confidence,
            is_deepfake: inferenceResult.is_deepfake,
          });

        } catch (error) {
          // Handle analysis error
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? ((error as unknown) as { response: { data?: { detail?: string } } }).response?.data?.detail || ((error as unknown) as Error).message || 'Analysis failed'
            : ((error as unknown) as Error).message || 'Analysis failed';
          
          await updateAnalysisJobWithResults(job.id, {
            status: 'failed',
            error_message: errorMessage,
          });

          if (userId) {
            // Send WebSocket error event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_FAILED',
              jobId,
              timestamp: Date.now(),
              data: {
                error: errorMessage,
              },
            });
          }

          logger.error(`[ANALYSIS FAILED] Job ${jobId} failed`, {
            correlationId,
            jobId,
            type,
            error: errorMessage,
          });
        }
      });

    } catch (error) {
      logger.error(`[ANALYSIS ERROR] ${type} analysis failed`, {
        correlationId,
        type,
        error: (error as Error).message,
      });
      
      return res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }
);

// Text analysis endpoint
router.post(
  '/text',
  validateRequest([]),
  async (req: AuthenticatedRequest, res: Response) => {
    const type = 'text';
    const correlationId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const userId = req.user?.id;
    const { text } = req.body;

    logger.info(`[ANALYSIS REQUEST] ${type} analysis started`, {
      correlationId,
      type,
      userId,
      textLength: text?.length || 0,
    });

    try {
      // Validate text input
      if (!text || typeof text !== 'string') {
        return res.status(400).json({
          success: false,
          message: 'Text content is required',
        });
      }

      if (text.trim().length === 0) {
        return res.status(400).json({
          success: false,
          message: 'Text content cannot be empty',
        });
      }

      if (text.length > 10000) {
        return res.status(400).json({
          success: false,
          message: 'Text content too long (maximum 10,000 characters)',
        });
      }

      // Create job with standardized ID
      const jobId = `job_${randomUUID()}`;
      
      // Create job in database
      const job = await createAnalysisJob(userId || '', {
        modality: type as 'image' | 'video' | 'audio' | 'text',
        filename: 'text-input',
        mime_type: 'text/plain',
        size_bytes: text.length,
        metadata: {
          textLength: text.length,
          wordCount: text.split(/\s+/).filter(word => word.length > 0).length,
        },
      });

      logger.info(`[ANALYSIS JOB] Created job ${jobId}`, {
        correlationId,
        jobId,
        type,
      });

      // Send immediate response with job_id
      res.status(202).json({
        success: true,
        job_id: jobId,
        status: 'processing',
      });

      // Process in background
      setImmediate(async () => {
        try {
          // Send progress update
          if (userId) {
            sendJobProgress(userId, jobId, 'analyzing', 30);
          }

          // Forward to Python service for text analysis
          const pythonResponse = await pythonBridge.post(`/api/v2/analysis/${type}`, {
            text,
            job_id: jobId,
            user_id: userId || '',
          }, {
            timeout: 60000, // 1 minute for text analysis
          }) as {
            data: {
              success: boolean;
              is_ai_generated: boolean;
              confidence: number;
              explanation: string;
              model_name: string;
            };
          };

          if (userId) {
            sendJobProgress(userId, jobId, 'processing', 70);
          }

          // Python returns text analysis result
          const textResult = pythonResponse.data;
          
          // Node owns the job lifecycle - update with results
          await updateAnalysisJobWithResults(job.id, {
            status: 'completed',
            confidence: textResult.confidence,
            is_deepfake: textResult.is_ai_generated,
            model_name: textResult.model_name,
            model_version: '1.0.0',
            analysis_data: {
              explanation: textResult.explanation,
              text_length: text.length,
            },
            proof_json: {
              analysis_type: 'text',
              timestamp: new Date().toISOString(),
            },
          });

          if (userId) {
            sendJobProgress(userId, jobId, 'completed', 100);

            // Send WebSocket event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_COMPLETED',
              jobId,
              timestamp: Date.now(),
              data: {
                status: 'completed',
                confidence: textResult.confidence,
                is_ai_generated: textResult.is_ai_generated,
                model_name: textResult.model_name,
                explanation: textResult.explanation,
              },
            });
          }

          logger.info(`[ANALYSIS COMPLETED] Job ${jobId} finished successfully`, {
            correlationId,
            jobId,
            type,
            confidence: textResult.confidence,
            is_ai_generated: textResult.is_ai_generated,
          });

        } catch (error) {
          // Handle analysis error
          const errorMessage = error && typeof error === 'object' && 'response' in error 
            ? ((error as unknown) as { response: { data?: { detail?: string } } }).response?.data?.detail || ((error as unknown) as Error).message || 'Analysis failed'
            : ((error as unknown) as Error).message || 'Analysis failed';
          
          await updateAnalysisJobWithResults(job.id, {
            status: 'failed',
            error_message: errorMessage,
          });

          if (userId) {
            // Send WebSocket error event
            webSocketManager.sendEventToUser(userId, {
              type: 'JOB_FAILED',
              jobId,
              timestamp: Date.now(),
              data: {
                error: errorMessage,
              },
            });
          }

          logger.error(`[ANALYSIS FAILED] Job ${jobId} failed`, {
            correlationId,
            jobId,
            type,
            error: errorMessage,
          });
        }
      });

    } catch (error) {
      logger.error(`[ANALYSIS ERROR] ${type} analysis failed`, {
        correlationId,
        type,
        error: (error as Error).message,
      });
      
      return res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }
);

// DISABLED: Multimodal analysis endpoint - temporarily deactivated
router.post(
  '/multimodal',
  upload.single('file'),
  validateRequest([]),
  async (req: AuthenticatedRequest, res: Response) => {
    return res.status(410).json({
      success: false,
      message: "Multimodal analysis is temporarily disabled"
    });
  }
);

// Get analysis status
router.get('/status/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const result = await pythonBridge.get<{
      success: boolean;
      data: {
        status: string;
        progress?: number;
        message?: string;
        created_at?: string;
        updated_at?: string;
        completed_at?: string;
        error?: string;
      };
    }>(`/analyze/status/${id}`);

    res.json({
      success: true,
      data: result.data,
    });
  } catch (error) {
    await handleAnalysisError(error, res);
  }
});

// Get analysis results (for polling)
router.get('/results/:jobId', async (req: Request, res: Response) => {
  try {
    const { jobId } = req.params;
    
    // Query the database for the job
    const result = await pythonBridge.get(`/results/${jobId}`) as {
      status: string;
      result?: {
        is_deepfake: boolean;
        confidence: number;
        model_name: string;
        model_version: string;
        analysis_data?: {
          processing_time?: number;
        };
      };
      error?: string;
    };
    
    // Transform the result to match the expected format
    const transformedResult = {
      id: jobId,
      status: result.status || 'processing',
      result: result.result ? {
        isAuthentic: !result.result.is_deepfake,
        confidence: result.result.confidence,
        details: {
          isDeepfake: result.result.is_deepfake,
          modelInfo: {
            name: result.result.model_name,
            version: result.result.model_version
          }
        },
        metrics: {
          processingTime: result.result.analysis_data?.processing_time || 0,
          modelVersion: result.result.model_version
        }
      } : undefined,
      error: result.error
    };
    
    res.json(transformedResult);
  } catch (error) {
    await handleAnalysisError(error, res);
  }
});

export { router as analysisRouter };

