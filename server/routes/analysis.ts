import { Router } from 'express';
import { pythonBridge } from '../services/python-bridge';
import { authenticateJWT } from '../middleware/auth';
import { v4 as uuidv4 } from 'uuid';
import multer from 'multer';
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
  authenticateJWT,
  upload.single('file'),
  async (req, res, next) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const fileType = getFileType(req.file.mimetype);
      const analysisId = uuidv4();
      
      // In a production environment, you might want to move the file to persistent storage
      // For now, we'll use the temporary upload location
      const filePath = req.file.path;

      try {
        const analysisResult = await pythonBridge.analyzeMedia({
          filePath,
          fileType,
          userId: req.user.id,
          metadata: {
            originalName: req.file.originalname,
            mimeType: req.file.mimetype,
            size: req.file.size,
          },
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
router.get('/analysis/status/:id', authenticateJWT, async (req, res, next) => {
  try {
    const { id } = req.params;
    const status = await pythonBridge.getAnalysisStatus(id);
    res.json(status);
  } catch (error) {
    next(error);
  }
});

// Get analysis history for the authenticated user
router.get('/analysis/history', authenticateJWT, async (req, res, next) => {
  try {
    // In a real implementation, you would fetch this from your database
    res.json({
      results: [],
      total: 0,
      page: 1,
      limit: 10,
    });
  } catch (error) {
    next(error);
  }
});

export default router;
