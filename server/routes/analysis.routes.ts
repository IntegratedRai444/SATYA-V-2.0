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

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/webm'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  },
});

// Helper function to handle analysis errors
const handleAnalysisError = (error: unknown, res: Response) => {
  const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
  logger.error('Analysis error:', error);
  return res.status(400).json({
    success: false,
    error: errorMessage,
  });
};

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
