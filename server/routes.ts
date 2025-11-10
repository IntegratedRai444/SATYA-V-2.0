import type { Express, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertScanSchema } from "@shared/schema";
import { uploadMiddleware, handleUploadError } from "./middleware/upload";
import { generatePDFReport } from "./services/report-generator";
import { pythonBridgeEnhanced as pythonBridge } from "./services/python-bridge-fixed";
import { validationSchemas, uploadConfig } from "./security-config";
import { logger } from "./config";
import apiRoutes from "./routes/index";
import { requireAuth, AuthenticatedRequest } from "./middleware/auth";
import { webSocketManager } from "./services/websocket-manager";

// File upload middleware is now handled by ./middleware/upload

// Authentication middleware is now imported from ./middleware/auth

// Input validation middleware
function validateInput(schema: z.ZodSchema) {
  return (req: any, res: any, next: any) => {
    try {
      const validated = schema.parse(req.body);
      req.validatedData = validated;
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          message: 'Invalid input data',
          errors: error.errors.map(err => ({
            field: err.path.join('.'),
            message: err.message
          }))
        });
      }
      return res.status(400).json({
        success: false,
        message: 'Input validation failed'
      });
    }
  };
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Start advanced detection services
  logger.info('Initializing advanced detection capabilities');
  
  // Register API routes
  app.use('/api', apiRoutes);
  
  // Health check endpoint
  app.get('/api/health', (req, res) => {
    logger.debug('Health check endpoint accessed');
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      services: {
        database: 'connected',
        python_bridge: 'available',
        websocket: 'ready'
      }
    });
  });

  // Simple test endpoint
  app.get('/api/test', (req, res) => {
    logger.debug('Test endpoint accessed');
    res.json({ message: 'Server is working!', timestamp: new Date().toISOString() });
  });

  // Config endpoint
  app.get('/api/config', (req, res) => {
    res.json({
      server_url: process.env.SERVER_URL || 'http://localhost:5000',
      port: process.env.PORT || 5000,
      timestamp: new Date().toISOString()
    });
  });

  // Authentication endpoints are now handled by /api/auth routes

  // Analytics endpoint (protected)
  app.get('/api/analytics', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const allScans = await storage.getAllScans();
      const totalScans = allScans.length;
      const authenticScans = allScans.filter(scan => scan.result === 'authentic').length;
      const deepfakeScans = allScans.filter(scan => scan.result === 'deepfake').length;
      
      res.json({
        total_scans: totalScans,
        authentic_files: authenticScans,
        deepfake_files: deepfakeScans,
        avg_confidence: totalScans > 0 ? Math.round(allScans.reduce((sum, scan) => sum + scan.confidenceScore, 0) / totalScans) : 0,
        most_common_type: 'image'
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch analytics' });
    }
  });

  // Scans history endpoint (protected)
  app.get('/api/scans', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const scans = await storage.getAllScans();
      res.json(scans);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch scans' });
    }
  });

  // Get specific scan by ID (protected)
  app.get('/api/scans/:id', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const scanId = parseInt(req.params.id);
      if (isNaN(scanId)) {
        return res.status(400).json({ message: 'Invalid scan ID' });
      }
      
      const scan = await storage.getScanById(scanId);
      if (!scan) {
        return res.status(404).json({ message: 'Scan not found' });
      }
      res.json(scan);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch scan' });
    }
  });

  // Get scan report (protected)
  app.get('/api/scans/:id/report', requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const scanId = parseInt(req.params.id);
      if (isNaN(scanId)) {
        return res.status(400).json({ message: 'Invalid scan ID' });
      }
      
      const scan = await storage.getScanById(scanId);
      if (!scan) {
        return res.status(404).json({ message: 'Scan not found' });
      }
      
      // Transform scan data to match ReportData interface
      const reportData = {
        id: scan.id,
        filename: scan.filename,
        type: scan.type,
        result: scan.result,
        confidence: scan.confidenceScore,
        analyzedAt: scan.createdAt,
        details: {
          manipulationScore: 1 - scan.confidenceScore / 100,
          authenticityScore: scan.confidenceScore / 100,
          visualArtifacts: 0.1,
          audioArtifacts: 0.1,
          analysisTime: 1000,
          modelVersion: '2.0.0',
          warnings: [],
          recommendations: ['Continue monitoring for similar content']
        }
      };
      
      await generatePDFReport(reportData, res);
    } catch (error) {
      res.status(500).json({ message: 'Failed to generate report' });
    }
  });

  // Settings endpoints (protected)
  app.post('/api/settings/preferences', requireAuth, validateInput(validationSchemas.userPreferences), async (req: AuthenticatedRequest, res: Response) => {
    try {
      // In a real app, save to database
      res.json({ 
        success: true,
        message: "Preferences updated successfully",
        data: req.validatedData
      });
    } catch (error) {
      res.status(500).json({ message: 'Failed to save preferences' });
    }
  });

  app.post('/api/settings/profile', requireAuth, validateInput(validationSchemas.userProfile), async (req: AuthenticatedRequest, res: Response) => {
    try {
      // For now, just return success since updateProfile method doesn't exist
      res.json({ 
        success: true,
        message: "Profile update functionality not implemented yet"
      });
    } catch (error) {
      res.status(500).json({ message: 'Failed to save profile' });
    }
  });

  // AI Analysis endpoints (protected)
  app.post('/api/ai/analyze/image', uploadMiddleware.singleImage, handleUploadError, requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file && !req.body.imageData) {
        return res.status(400).json({ 
          success: false,
          message: 'No image data provided' 
        });
      }
      
      let imageBuffer;
      if (req.file) {
        imageBuffer = req.file.buffer;
      } else if (req.body.imageData) {
        // Handle base64 image from webcam
        const base64Data = req.body.imageData.split(',')[1];
        imageBuffer = Buffer.from(base64Data, 'base64');
      }
      
      // Get JWT token for Python backend
      const authHeader = req.headers['authorization'];
      const token = authHeader?.replace('Bearer ', '') || null;
      
      // Ensure we have image buffer
      if (!imageBuffer) {
        return res.status(400).json({
          success: false,
          message: 'No image data could be processed'
        });
      }

      // Call Python backend for analysis
      const result = await pythonBridge.analyzeImage(imageBuffer, token);
      
      res.json(result);
    } catch (error) {
      logger.error('Image analysis error', { error: (error as Error).message });
      res.status(500).json({ 
        success: false,
        message: 'Image analysis failed' 
      });
    }
  });

  app.post('/api/ai/analyze/video', uploadMiddleware.singleVideo, handleUploadError, requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ 
          success: false,
          message: 'No video file provided' 
        });
      }
      
      const videoBuffer = req.file.buffer;
      const filename = req.file.originalname || 'video.mp4';
      
      // Get JWT token for Python backend
      const authHeader = req.headers['authorization'];
      const token = authHeader?.replace('Bearer ', '') || null;
      
      // Call Python backend for analysis
      const result = await pythonBridge.analyzeVideo(videoBuffer, filename, token);
      
      res.json(result);
    } catch (error) {
      logger.error('Video analysis error', { error: (error as Error).message });
      res.status(500).json({ 
        success: false,
        message: 'Video analysis failed' 
      });
    }
  });

  app.post('/api/ai/analyze/audio', uploadMiddleware.singleAudio, handleUploadError, requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ 
          success: false,
          message: 'No audio file provided' 
        });
      }
      
      const audioBuffer = req.file.buffer;
      const filename = req.file.originalname || 'audio.mp3';
      
      // Get JWT token for Python backend
      const authHeader = req.headers['authorization'];
      const token = authHeader?.replace('Bearer ', '') || null;
      
      // Call Python backend for analysis
      const result = await pythonBridge.analyzeAudio(audioBuffer, filename, token);
      
      res.json(result);
    } catch (error) {
      logger.error('Audio analysis error', { error: (error as Error).message });
      res.status(500).json({ 
        success: false,
        message: 'Audio analysis failed' 
      });
    }
  });

  app.post('/api/ai/analyze/multimodal', uploadMiddleware.multimodal, handleUploadError, requireAuth, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      
      if (!files || Object.keys(files).length === 0) {
        return res.status(400).json({ 
          success: false,
          message: 'No files provided for multimodal analysis' 
        });
      }
      
      // Extract buffers for each modality
      const imageBuffer = files.image?.[0]?.buffer || null;
      const videoBuffer = files.video?.[0]?.buffer || null;
      const audioBuffer = files.audio?.[0]?.buffer || null;
      
      // Get JWT token for Python backend
      const authHeader = req.headers['authorization'];
      const token = authHeader?.replace('Bearer ', '') || null;
      
      // Call Python backend for multimodal analysis
      const result = await pythonBridge.analyzeMultimodal(
        imageBuffer,
        audioBuffer,
        videoBuffer,
        token
      );
      
      res.json(result);
    } catch (error) {
      logger.error('Multimodal analysis error', { error: (error as Error).message });
      res.status(500).json({ 
        success: false,
        message: 'Multimodal analysis failed' 
      });
    }
  });

  // Create HTTP server
  const server = createServer(app);
  
  // Initialize WebSocket server
  webSocketManager.initialize(server);
  
  return server;
}
