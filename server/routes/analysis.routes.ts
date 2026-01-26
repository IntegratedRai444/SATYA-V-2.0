import { Router, Request, Response } from 'express';
import multer from 'multer';
import type { Express } from 'express';
import path from 'path';
import { validateRequest } from '../middleware/validate-request';
import { pythonBridge } from '../services/python-http-bridge';
import { logger } from '../config/logger';
import { createAnalysisJob, updateAnalysisJobWithResults } from './history';
import FormData from 'form-data';
import { AuthenticatedRequest } from '../types/auth';

// Type definition for analysis result
interface AnalysisResult {
  status: string;
  confidence?: number;
  is_deepfake?: boolean;
  model_name?: string;
  model_version?: string;
  summary?: Record<string, unknown>;
  proof?: Record<string, unknown>;
  error?: string;
}

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
    magic: [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50],
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
  fileFilter: (req, file, cb) => {
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

      // Validate filename
      const sanitizedFilename = path.basename(file.originalname).replace(/[^a-zA-Z0-9\-_.]/g, '');
      if (sanitizedFilename !== file.originalname) {
        return cb(new Error('Invalid characters in filename'));
      }

      // Check for path traversal
      if (file.originalname.includes('..') || path.isAbsolute(file.originalname)) {
        return cb(new Error('Invalid file path'));
      }

      // For small files, we can check magic numbers immediately
      const chunks: Buffer[] = [];
      
      req.on('data', (chunk: Buffer) => {
        chunks.push(chunk);
        const buffer = Buffer.concat(chunks);
        
        if (buffer.length >= fileConfig.magic.length) {
          const magic = buffer.slice(0, fileConfig.magic.length);
          if (!checkMagicNumbers(magic, fileConfig.magic)) {
            return cb(new Error('Invalid file signature'));
          }
          // Remove the data listener once we've checked the magic numbers
          req.removeAllListeners('data');
        }
      });
      
      req.on('end', () => {
        // Handle any remaining data
        const buffer = Buffer.concat(chunks);
        if (buffer.length > 0 && buffer.length < fileConfig.magic.length) {
          return cb(new Error('File too small to determine type'));
        }
      });

      // Check file size against type-specific limit
      if (file.size > fileConfig.maxSize) {
        return cb(new Error(`File too large. Maximum size: ${formatBytes(fileConfig.maxSize)}`));
      }

      // Additional security checks
      if (file.originalname !== sanitizedFilename) {
        return cb(new Error('Invalid filename'));
      }

      cb(null, true);
    } catch (error) {
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

// Input validation middleware for analysis requests (defined but not used - available for future use)
/*
const validateAnalysisRequest = [
  // Check if file exists
  (req: Request, res: Response, next: NextFunction) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        code: 'NO_FILE_PROVIDED',
        message: 'No file was uploaded',
      });
    }
    next();
  },
  
  // Validate file size
  (req: Request, res: Response, next: NextFunction) => {
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
    if (req.file && req.file.size > MAX_FILE_SIZE) {
      return res.status(400).json({
        success: false,
        code: 'FILE_TOO_LARGE',
        message: `File size exceeds the limit of ${MAX_FILE_SIZE / (1024 * 1024)}MB`,
      });
    }
    next();
  },
  
  // Validate MIME type
  (req: Request, res: Response, next: NextFunction) => {
    if (req.file) {
      const allowedTypes = Object.keys(ALLOWED_FILE_TYPES);
      if (!allowedTypes.includes(req.file.mimetype)) {
        return res.status(400).json({
          success: false,
          code: 'INVALID_FILE_TYPE',
          message: `Invalid file type. Allowed types: ${allowedTypes.join(', ')}`,
        });
      }
    }
    next();
  }
];
*/

// Analyze image
router.post(
  '/image',
  upload.single('image'),
  async (req: AuthenticatedRequest, res: Response) => {
    let job: { id: string; report_code: string } | null = null;
    
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'User not authenticated' });
      }

      // Create analysis job in Supabase
      job = await createAnalysisJob(userId, {
        modality: 'image',
        filename: req.file.originalname,
        mime_type: req.file.mimetype,
        size_bytes: req.file.size,
        metadata: {
          originalName: req.file.originalname,
          uploadedAt: new Date().toISOString()
        }
      });

      // Process the image using Python service
      // Create FormData to match Python's UploadFile expectation
      const form = new FormData();
      
      // Append the file buffer as a file with proper filename and mimetype
      form.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      
      // Add additional metadata as form fields
      form.append('jobId', job.id);
      form.append('analyze_forensics', 'true');

      const result = await (pythonBridge as unknown as { request: (options: unknown) => Promise<unknown> }).request({
        method: 'POST',
        url: '/api/v2/analysis/unified/image',
        data: form,
        headers: {
          ...form.getHeaders()
        }
      }) as { data: { success: boolean; result: { success: boolean; authenticity?: string; confidence?: number; is_deepfake?: boolean; model_name?: string; model_version?: string; summary?: Record<string, unknown>; proof?: Record<string, unknown>; error?: string } } };

      // Save results to Supabase
      if (result.data && result.data.success) {
        const analysisResult = result.data.result;
        await updateAnalysisJobWithResults(job.id, {
          status: 'completed',
          confidence: analysisResult.confidence || 0,
          is_deepfake: analysisResult.is_deepfake || false,
          model_name: analysisResult.model_name || 'SatyaAI-Image',
          model_version: analysisResult.model_version || '1.0.0',
          summary: analysisResult.summary || {},
          analysis_data: analysisResult,
          proof_json: analysisResult.proof || {}
        });

        res.json({
          success: true,
          data: {
            ...result.data,
            jobId: job.id,
            reportCode: job.report_code
          }
        });
      } else {
        // Mark job as failed
        await updateAnalysisJobWithResults(job.id, {
          status: 'failed',
          error_message: result.data?.result?.error || 'Analysis failed'
        });

        res.status(500).json({
          success: false,
          error: result.data?.result?.error || 'Analysis failed'
        });
      }
    } catch (error) {
      await handleAnalysisError(error, res, job?.id);
    }
  }
);

// Analyze audio
router.post(
  '/audio',
  upload.single('audio'),
  async (req: AuthenticatedRequest, res: Response) => {
    let job: { id: string; report_code: string } | null = null;
    
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'User not authenticated' });
      }

      // Create analysis job in Supabase
      job = await createAnalysisJob(userId, {
        modality: 'audio',
        filename: req.file.originalname,
        mime_type: req.file.mimetype,
        size_bytes: req.file.size,
        metadata: {
          originalName: req.file.originalname,
          uploadedAt: new Date().toISOString()
        }
      });

      // Process the audio using Python service
      // Create FormData to match Python's UploadFile expectation
      const form = new FormData();
      
      // Append the file buffer as a file with proper filename and mimetype
      form.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      
      // Add additional metadata as form fields
      form.append('jobId', job.id);
      form.append('analyze_forensics', 'true');

      const result = await (pythonBridge as unknown as { request: (options: unknown) => Promise<unknown> }).request({
        method: 'POST',
        url: '/api/v2/analysis/unified/audio',
        data: form,
        headers: {
          ...form.getHeaders()
        }
      }) as { data: { success: boolean; result: { success: boolean; authenticity?: string; confidence?: number; is_deepfake?: boolean; model_name?: string; model_version?: string; summary?: Record<string, unknown>; proof?: Record<string, unknown>; error?: string } } };

      // Save results to Supabase
      if (result.data && result.data.success) {
        const analysisResult = result.data.result;
        await updateAnalysisJobWithResults(job.id, {
          status: 'completed',
          confidence: analysisResult.confidence || 0,
          is_deepfake: analysisResult.is_deepfake || false,
          model_name: analysisResult.model_name || 'SatyaAI-Audio',
          model_version: analysisResult.model_version || '1.0.0',
          summary: analysisResult.summary || {},
          analysis_data: analysisResult,
          proof_json: analysisResult.proof || {}
        });

        res.json({
          success: true,
          data: {
            ...result.data,
            jobId: job.id,
            reportCode: job.report_code
          }
        });
      } else {
        // Mark job as failed
        await updateAnalysisJobWithResults(job.id, {
          status: 'failed',
          error_message: result.data?.result?.error || 'Analysis failed'
        });

        res.status(500).json({
          success: false,
          error: result.data?.result?.error || 'Analysis failed'
        });
      }
    } catch (error) {
      await handleAnalysisError(error, res, job?.id);
    }
  }
);

// Batch analysis
router.post(
  '/batch',
  upload.array('files', 10), // Max 10 files
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
      }

      // Process each file
      const results = await Promise.all(
        (req.files as Express.Multer.File[]).map(async (file) => {
          try {
            const result = await (pythonBridge as unknown as { post: (url: string, data: unknown) => Promise<unknown> }).post('/analyze/batch', {
              file: file.buffer.toString('base64'),
              mimeType: file.mimetype,
              originalname: file.originalname,
            });
            return {
              filename: file.originalname,
              success: true,
              data: (result as { data: unknown }).data,
            };
          } catch (error) {
            return {
              filename: file.originalname,
              success: false,
              error: error instanceof Error ? error.message : 'Analysis failed',
            };
          }
        })
      );

      res.json({
        success: true,
        results,
      });
    } catch (error) {
      await handleAnalysisError(error, res);
    }
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

// Analyze video
router.post(
  '/video',
  upload.single('video'),
  validateRequest,
  async (req: AuthenticatedRequest, res: Response) => {
    let job: { id: string; report_code: string } | null = null;
    
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: 'Video file is required'
        });
      }

      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }

      // Create analysis job
      job = await createAnalysisJob(userId, {
        modality: 'video',
        filename: req.file.originalname,
        mime_type: req.file.mimetype,
        size_bytes: req.file.size,
        status: 'pending'
      });

      // Process the video using Python service
      // Create FormData to match Python's UploadFile expectation
      const form = new FormData();
      
      // Append the file buffer as a file with proper filename and mimetype
      form.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      
      // Add additional metadata as form fields
      form.append('jobId', job.id);
      form.append('analyze_forensics', 'true');

      const result = await (pythonBridge as unknown as { request: (options: unknown) => Promise<unknown> }).request({
        method: 'POST',
        url: '/api/v2/analysis/unified/video',
        data: form,
        headers: {
          ...form.getHeaders()
        }
      }) as { data: { success: boolean; result: { success: boolean; authenticity?: string; confidence?: number; is_deepfake?: boolean; model_name?: string; model_version?: string; summary?: Record<string, unknown>; proof?: Record<string, unknown>; error?: string } } };

      // Save results to Supabase
      if (result.data && result.data.success) {
        const analysisResult = result.data.result;
        await updateAnalysisJobWithResults(job.id, {
          status: 'completed',
          confidence: analysisResult.confidence || 0,
          is_deepfake: analysisResult.is_deepfake || false,
          model_name: analysisResult.model_name || 'SatyaAI-Video',
          model_version: analysisResult.model_version || '1.0.0',
          summary: analysisResult.summary || {},
          analysis_data: analysisResult,
          proof_json: analysisResult.proof || {}
        });

        res.json({
          success: true,
          data: {
            ...result.data,
            jobId: job.id,
            reportCode: job.report_code
          }
        });
      } else {
        // Mark job as failed
        await updateAnalysisJobWithResults(job.id, {
          status: 'failed',
          error_message: result.data?.result?.error || 'Analysis failed'
        });

        res.status(500).json({
          success: false,
          error: result.data?.result?.error || 'Analysis failed'
        });
      }
    } catch (error) {
      await handleAnalysisError(error, res, job?.id);
    }
  }
);

// Analyze multimodal
router.post(
  '/multimodal',
  upload.array('files', 5), // Max 5 files for multimodal
  validateRequest,
  async (req: AuthenticatedRequest, res: Response) => {
    let job: { id: string; report_code: string } | null = null;
    
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'At least one file is required for multimodal analysis'
        });
      }

      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }

      // Create analysis job
      job = await createAnalysisJob(userId, {
        modality: 'multimodal',
        filename: `${req.files.length} files`,
        mime_type: 'multimodal',
        size_bytes: (req.files as Express.Multer.File[]).reduce((total: number, file: Express.Multer.File) => total + file.size, 0),
        status: 'pending'
      });

      // Process using Python service
      // Create FormData to match Python's UploadFile expectation
      const form = new FormData();
      
      // Add each file as a separate form field
      (req.files as Express.Multer.File[]).forEach((file, index) => {
        form.append(`file${index}`, file.buffer, {
          filename: file.originalname,
          contentType: file.mimetype
        });
      });
      
      // Add metadata
      form.append('jobId', job.id);

      const result = await (pythonBridge as unknown as { request: (options: unknown) => Promise<unknown> }).request({
        method: 'POST',
        url: '/api/v2/analysis/multimodal',
        data: form,
        headers: {
          ...form.getHeaders()
        }
      }) as { data: { success: boolean; result: { success: boolean; authenticity?: string; confidence?: number; is_deepfake?: boolean; model_name?: string; model_version?: string; summary?: Record<string, unknown>; proof?: Record<string, unknown>; error?: string } } };

      // Save results to Supabase
      if (result.data && result.data.success) {
        const analysisResult = result.data.result;
        await updateAnalysisJobWithResults(job.id, {
          status: 'completed',
          confidence: analysisResult.confidence || 0,
          is_deepfake: analysisResult.is_deepfake || false,
          model_name: analysisResult.model_name || 'SatyaAI-Multimodal',
          model_version: analysisResult.model_version || '1.0.0',
          summary: analysisResult.summary || {},
          analysis_data: analysisResult,
          proof_json: analysisResult.proof || {}
        });

        res.json({
          success: true,
          data: {
            ...result.data,
            jobId: job.id,
            reportCode: job.report_code
          }
        });
      } else {
        // Mark job as failed
        const errorMessage = result.data?.result?.error || result.data?.error || 'Analysis failed';
        await updateAnalysisJobWithResults(job.id, {
          status: 'failed',
          error_message: errorMessage
        });

        res.status(500).json({
          success: false,
          error: errorMessage
        });
      }
    } catch (error) {
      await handleAnalysisError(error, res, job?.id);
    }
  }
);

// Analyze webcam capture
router.post(
  '/webcam',
  upload.single('image'),
  validateRequest,
  async (req: AuthenticatedRequest, res: Response) => {
    let job: { id: string; report_code: string } | null = null;
    
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: 'Webcam image is required'
        });
      }

      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }

      // Create analysis job
      job = await createAnalysisJob(userId, {
        modality: 'webcam',
        filename: `webcam-${Date.now()}.jpg`,
        mime_type: req.file.mimetype,
        size_bytes: req.file.size,
        status: 'pending'
      });

      // Process the webcam image using Python service
      const result = await (pythonBridge as unknown as { request: (options: unknown) => Promise<unknown> }).request({
        method: 'POST',
        url: '/analysis/webcam',
        data: {
          image: req.file.buffer.toString('base64'),
          mimeType: req.file.mimetype,
          jobId: job.id,
          filename: req.file.originalname,
        },
      }) as { data: { status: string; confidence?: number; is_deepfake?: boolean; model_name?: string; model_version?: string; summary?: Record<string, unknown>; proof?: Record<string, unknown>; error?: string } };

      // Save results to Supabase
      if (result.data && (result.data as AnalysisResult).status === 'success') {
        await updateAnalysisJobWithResults(job.id, {
          status: 'completed',
          confidence: (result.data as AnalysisResult).confidence || 0,
          is_deepfake: (result.data as AnalysisResult).is_deepfake || false,
          model_name: (result.data as AnalysisResult).model_name || 'SatyaAI-Webcam',
          model_version: (result.data as AnalysisResult).model_version || '1.0.0',
          summary: (result.data as AnalysisResult).summary || {},
          analysis_data: result.data,
          proof_json: (result.data as AnalysisResult).proof || {}
        });

        res.json({
          success: true,
          data: {
            ...result.data,
            jobId: job.id,
            reportCode: job.report_code
          }
        });
      } else {
        // Mark job as failed
        await updateAnalysisJobWithResults(job.id, {
          status: 'failed',
          error_message: (result.data as AnalysisResult)?.error || 'Analysis failed'
        });

        res.status(500).json({
          success: false,
          error: (result.data as AnalysisResult)?.error || 'Analysis failed'
        });
      }
    } catch (error) {
      await handleAnalysisError(error, res, job?.id);
    }
  }
);

export { router as analysisRouter };
