import { Router, Response } from 'express';
import { requireAuth, AuthenticatedRequest } from '../middleware/auth';
import { 
  uploadMiddleware, 
  validateUploadedFiles, 
  cleanupOnError, 
  handleUploadError 
} from '../middleware/upload';
import { logger } from '../config';
import path from 'path';
import fs from 'fs/promises';

const router = Router();

/**
 * POST /api/upload/image
 * Upload single image file
 */
router.post('/image', 
  requireAuth,
  cleanupOnError,
  uploadMiddleware.singleImage,
  handleUploadError,
  validateUploadedFiles,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          message: 'No image file provided',
          code: 'NO_FILE'
        });
      }

      const fileInfo = {
        id: generateFileId(),
        originalName: req.file.originalname,
        filename: req.file.filename,
        path: req.file.path,
        size: req.file.size,
        mimeType: req.file.mimetype,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.user?.userId
      };

      logger.info('Image uploaded successfully', {
        userId: req.user?.userId,
        filename: req.file.filename,
        size: req.file.size,
        mimeType: req.file.mimetype
      });

      res.json({
        success: true,
        message: 'Image uploaded successfully',
        file: fileInfo
      });
    } catch (error) {
      logger.error('Image upload processing error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to process uploaded image',
        code: 'PROCESSING_ERROR'
      });
    }
  }
);

/**
 * POST /api/upload/video
 * Upload single video file
 */
router.post('/video',
  requireAuth,
  cleanupOnError,
  uploadMiddleware.singleVideo,
  handleUploadError,
  validateUploadedFiles,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          message: 'No video file provided',
          code: 'NO_FILE'
        });
      }

      const fileInfo = {
        id: generateFileId(),
        originalName: req.file.originalname,
        filename: req.file.filename,
        path: req.file.path,
        size: req.file.size,
        mimeType: req.file.mimetype,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.user?.userId
      };

      logger.info('Video uploaded successfully', {
        userId: req.user?.userId,
        filename: req.file.filename,
        size: req.file.size,
        mimeType: req.file.mimetype
      });

      res.json({
        success: true,
        message: 'Video uploaded successfully',
        file: fileInfo
      });
    } catch (error) {
      logger.error('Video upload processing error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to process uploaded video',
        code: 'PROCESSING_ERROR'
      });
    }
  }
);

/**
 * POST /api/upload/audio
 * Upload single audio file
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

      const fileInfo = {
        id: generateFileId(),
        originalName: req.file.originalname,
        filename: req.file.filename,
        path: req.file.path,
        size: req.file.size,
        mimeType: req.file.mimetype,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.user?.userId
      };

      logger.info('Audio uploaded successfully', {
        userId: req.user?.userId,
        filename: req.file.filename,
        size: req.file.size,
        mimeType: req.file.mimetype
      });

      res.json({
        success: true,
        message: 'Audio uploaded successfully',
        file: fileInfo
      });
    } catch (error) {
      logger.error('Audio upload processing error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to process uploaded audio',
        code: 'PROCESSING_ERROR'
      });
    }
  }
);

/**
 * POST /api/upload/multiple
 * Upload multiple files
 */
router.post('/multiple',
  requireAuth,
  cleanupOnError,
  uploadMiddleware.multipleFiles,
  handleUploadError,
  validateUploadedFiles,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const files = req.files as Express.Multer.File[];
      
      if (!files || files.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files provided',
          code: 'NO_FILES'
        });
      }

      const fileInfos = files.map(file => ({
        id: generateFileId(),
        originalName: file.originalname,
        filename: file.filename,
        path: file.path,
        size: file.size,
        mimeType: file.mimetype,
        uploadedAt: new Date().toISOString(),
        uploadedBy: req.user?.userId
      }));

      const totalSize = files.reduce((sum, file) => sum + file.size, 0);

      logger.info('Multiple files uploaded successfully', {
        userId: req.user?.userId,
        fileCount: files.length,
        totalSize,
        files: files.map(f => ({ name: f.filename, size: f.size, type: f.mimetype }))
      });

      res.json({
        success: true,
        message: `${files.length} files uploaded successfully`,
        files: fileInfos,
        totalSize
      });
    } catch (error) {
      logger.error('Multiple files upload processing error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to process uploaded files',
        code: 'PROCESSING_ERROR'
      });
    }
  }
);

/**
 * POST /api/upload/multimodal
 * Upload files for multimodal analysis (image, video, audio)
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

      const result: any = {
        success: true,
        message: 'Files uploaded successfully for multimodal analysis',
        files: {}
      };

      let totalSize = 0;
      let fileCount = 0;

      // Process each file type
      for (const [fieldName, fileArray] of Object.entries(files)) {
        if (fileArray && fileArray.length > 0) {
          const file = fileArray[0]; // Take first file for each type
          
          result.files[fieldName] = {
            id: generateFileId(),
            originalName: file.originalname,
            filename: file.filename,
            path: file.path,
            size: file.size,
            mimeType: file.mimetype,
            uploadedAt: new Date().toISOString(),
            uploadedBy: req.user?.userId
          };

          totalSize += file.size;
          fileCount++;
        }
      }

      result.totalSize = totalSize;
      result.fileCount = fileCount;

      logger.info('Multimodal files uploaded successfully', {
        userId: req.user?.userId,
        fileTypes: Object.keys(result.files),
        fileCount,
        totalSize
      });

      res.json(result);
    } catch (error) {
      logger.error('Multimodal upload processing error', {
        error: (error as Error).message,
        userId: req.user?.userId
      });

      res.status(500).json({
        success: false,
        message: 'Failed to process multimodal upload',
        code: 'PROCESSING_ERROR'
      });
    }
  }
);

/**
 * DELETE /api/upload/:filename
 * Delete uploaded file
 */
router.delete('/:filename',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { filename } = req.params;
      
      // Validate filename (security check)
      if (!filename || filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
        return res.status(400).json({
          success: false,
          message: 'Invalid filename',
          code: 'INVALID_FILENAME'
        });
      }

      const filePath = path.join(process.cwd(), 'uploads', filename);
      
      try {
        await fs.access(filePath);
        await fs.unlink(filePath);
        
        logger.info('File deleted successfully', {
          userId: req.user?.userId,
          filename,
          filePath
        });

        res.json({
          success: true,
          message: 'File deleted successfully'
        });
      } catch (error) {
        if ((error as any).code === 'ENOENT') {
          return res.status(404).json({
            success: false,
            message: 'File not found',
            code: 'FILE_NOT_FOUND'
          });
        }
        throw error;
      }
    } catch (error) {
      logger.error('File deletion error', {
        error: (error as Error).message,
        userId: req.user?.userId,
        filename: req.params.filename
      });

      res.status(500).json({
        success: false,
        message: 'Failed to delete file',
        code: 'DELETION_ERROR'
      });
    }
  }
);

/**
 * GET /api/upload/info/:filename
 * Get file information
 */
router.get('/info/:filename',
  requireAuth,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { filename } = req.params;
      
      // Validate filename (security check)
      if (!filename || filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
        return res.status(400).json({
          success: false,
          message: 'Invalid filename',
          code: 'INVALID_FILENAME'
        });
      }

      const filePath = path.join(process.cwd(), 'uploads', filename);
      
      try {
        const stats = await fs.stat(filePath);
        
        res.json({
          success: true,
          file: {
            filename,
            size: stats.size,
            createdAt: stats.birthtime,
            modifiedAt: stats.mtime,
            isFile: stats.isFile(),
            isDirectory: stats.isDirectory()
          }
        });
      } catch (error) {
        if ((error as any).code === 'ENOENT') {
          return res.status(404).json({
            success: false,
            message: 'File not found',
            code: 'FILE_NOT_FOUND'
          });
        }
        throw error;
      }
    } catch (error) {
      logger.error('File info retrieval error', {
        error: (error as Error).message,
        userId: req.user?.userId,
        filename: req.params.filename
      });

      res.status(500).json({
        success: false,
        message: 'Failed to retrieve file information',
        code: 'INFO_ERROR'
      });
    }
  }
);

/**
 * Generate unique file ID
 */
function generateFileId(): string {
  return `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export default router;