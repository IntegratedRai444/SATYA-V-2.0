import { Router, Request, Response, NextFunction } from 'express';
import { body, type ValidationChain } from 'express-validator';
import multer from 'multer';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import { promisify } from 'util';
import { validateRequest } from '../middleware/validate-request';
import { authenticate } from '../middleware/auth.middleware';
import { pythonBridge } from '../services/python-http-bridge';
import { logger } from '../config/logger';

const router = Router();
const writeFile = promisify(fs.writeFile);

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
const handleAnalysisError = (error: unknown, res: Response, context: Record<string, any> = {}) => {
  // Create a safe error object
  const safeError = error instanceof Error 
    ? { 
        name: error.name,
        message: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      }
    : { message: 'An unknown error occurred' };
  
  // Log the error with context (redacting sensitive data)
  logger.error('Analysis error', {
    ...safeError,
    ...context,
    // Redact sensitive data from context
    headers: undefined,
    body: undefined,
    file: context.file ? '[REDACTED]' : undefined,
  });

  // Return a sanitized error response
  return res.status(400).json({
    success: false,
    code: 'ANALYSIS_ERROR',
    message: 'Analysis failed. Please try again with a different file.',
    // Only include error details in development
    ...(process.env.NODE_ENV === 'development' && { error: safeError.message }),
    // Include a request ID for support
    requestId: res.locals.requestId,
  });
};

// Input validation middleware for analysis requests
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

// Analyze image
router.post(
  '/image',
  authenticate,
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Process the image using the Python service
      const result = await (pythonBridge as any).request({
        method: 'POST',
        url: '/analyze/image',
        data: {
          image: req.file.buffer.toString('base64'),
          mimeType: req.file.mimetype,
        },
      });

      res.json({
        success: true,
        data: result.data,
      });
    } catch (error) {
      handleAnalysisError(error, res);
    }
  }
);

// Analyze video
router.post(
  '/video',
  authenticate,
  upload.single('video'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Process the video using the Python service
      const result = await (pythonBridge as any).request({
        method: 'POST',
        url: '/analyze/video',
        data: {
          video: req.file.buffer.toString('base64'),
          mimeType: req.file.mimetype,
        },
      });

      res.json({
        success: true,
        data: result.data,
      });
    } catch (error) {
      handleAnalysisError(error, res);
    }
  }
);

// Analyze audio
router.post(
  '/audio',
  authenticate,
  upload.single('audio'),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Process the audio using the Python service
      const result = await pythonBridge.post('/analyze/audio', {
        audio: req.file.buffer.toString('base64'),
        mimeType: req.file.mimetype,
      });

      res.json({
        success: true,
        data: result.data,
      });
    } catch (error) {
      handleAnalysisError(error, res);
    }
  }
);

// Batch analysis
router.post(
  '/batch',
  authenticate,
  upload.array('files', 10), // Max 10 files
  async (req: Request, res: Response) => {
    try {
      if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
      }

      // Process each file
      const results = await Promise.all(
        (req.files as Express.Multer.File[]).map(async (file) => {
          try {
            const result = await pythonBridge.post('/analyze/batch', {
              file: file.buffer.toString('base64'),
              mimeType: file.mimetype,
              originalname: file.originalname,
            });
            return {
              filename: file.originalname,
              success: true,
              data: result.data,
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
      handleAnalysisError(error, res);
    }
  }
);

// Get analysis status
router.get('/status/:id', authenticate, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const result = await pythonBridge.get(`/analyze/status/${id}`);

    res.json({
      success: true,
      data: result.data,
    });
  } catch (error) {
    handleAnalysisError(error, res);
  }
});

export { router as analysisRouter };
