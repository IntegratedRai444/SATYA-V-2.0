import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertScanSchema } from "@shared/schema";
import multer from "multer";
import { deepfakeDetector } from "./services/deepfake-detector";
import { advancedDeepfakeDetector } from "./services/advanced-deepfake-detector";

// Setup multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max file size
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Set up API routes
  const apiRouter = app.route('/api');

  // Get recent scans
  app.get('/api/scans/recent', async (req, res) => {
    try {
      const recentScans = await storage.getRecentScans();
      res.json(recentScans);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch recent scans' });
    }
  });

  // Get all scans
  app.get('/api/scans', async (req, res) => {
    try {
      const scans = await storage.getAllScans();
      res.json(scans);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch scans' });
    }
  });

  // Get specific scan by ID
  app.get('/api/scans/:id', async (req, res) => {
    try {
      const scan = await storage.getScanById(parseInt(req.params.id));
      if (!scan) {
        return res.status(404).json({ message: 'Scan not found' });
      }
      res.json(scan);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch scan' });
    }
  });

  // Handle file upload and analysis
  app.post('/api/analyze', upload.array('media'), async (req, res) => {
    try {
      if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
        return res.status(400).json({ message: 'No files uploaded' });
      }

      const files = req.files as Express.Multer.File[];
      const file = files[0];
      const type = req.body.type || 'image';
      
      console.log(`Analyzing ${type} file: ${file.originalname}`);
      
      // Process file using the advanced deepfake detector service
      let detectionResult;
      if (type === 'image') {
        detectionResult = await advancedDeepfakeDetector.analyzeImage(file.buffer);
      } else if (type === 'video') {
        detectionResult = await advancedDeepfakeDetector.analyzeVideo(file.buffer);
      } else if (type === 'audio') {
        detectionResult = await advancedDeepfakeDetector.analyzeAudio(file.buffer);
      } else {
        return res.status(400).json({ message: 'Unsupported media type' });
      }
      
      // Convert detection result to scan record format
      const scanData = advancedDeepfakeDetector.convertToScanRecord(
        detectionResult, 
        file.originalname,
        type as 'image' | 'video' | 'audio'
      );
      
      // Save scan to database with user ID 1 (demo user)
      const result = await storage.createScan({
        userId: 1,
        ...scanData
      });

      res.json(result);
    } catch (error) {
      console.error('Analysis error:', error);
      res.status(500).json({ message: 'Analysis failed' });
    }
  });

  // Webcam analysis endpoint
  app.post('/api/analyze/webcam', async (req, res) => {
    try {
      // Get webcam image data from request body if available
      const imageData = req.body.imageData;
      let imageBuffer = Buffer.from([]);
      
      if (imageData && imageData.startsWith('data:image')) {
        // Extract base64 data from data URL
        const base64Data = imageData.split(',')[1];
        imageBuffer = Buffer.from(base64Data, 'base64');
      }
      
      // Process the webcam image with advanced detector
      const detectionResult = await advancedDeepfakeDetector.analyzeWebcam(imageBuffer);
      
      // Convert to scan record format
      const scanData = advancedDeepfakeDetector.convertToScanRecord(
        detectionResult, 
        'webcam_capture.jpg',
        'image'
      );
      
      // Save to database
      const result = await storage.createScan({
        userId: 1,
        ...scanData
      });

      res.json(result);
    } catch (error) {
      console.error('Webcam analysis error:', error);
      res.status(500).json({ message: 'Webcam analysis failed' });
    }
  });

  // Multimodal analysis endpoint (combines multiple inputs)
  app.post('/api/analyze/multimodal', upload.fields([
    { name: 'image', maxCount: 1 },
    { name: 'video', maxCount: 1 },
    { name: 'audio', maxCount: 1 }
  ]), async (req, res) => {
    try {
      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      
      // Get buffers for each type if available
      const imageBuffer = files.image?.[0]?.buffer;
      const videoBuffer = files.video?.[0]?.buffer;
      const audioBuffer = files.audio?.[0]?.buffer;
      
      if (!imageBuffer && !videoBuffer && !audioBuffer) {
        return res.status(400).json({ message: 'No media files provided for multimodal analysis' });
      }
      
      // Use the most advanced analysis method
      const detectionResult = await advancedDeepfakeDetector.analyzeMultimodal(
        imageBuffer, 
        audioBuffer,
        videoBuffer
      );
      
      // Determine primary file type for record
      let primaryType: 'image' | 'video' | 'audio' = 'image';
      let filename = 'multimodal_analysis.json';
      
      if (files.video?.length) {
        primaryType = 'video';
        filename = files.video[0].originalname;
      } else if (files.image?.length) {
        primaryType = 'image';
        filename = files.image[0].originalname;
      } else if (files.audio?.length) {
        primaryType = 'audio';
        filename = files.audio[0].originalname;
      }
      
      // Convert to scan record
      const scanData = advancedDeepfakeDetector.convertToScanRecord(
        detectionResult,
        filename,
        primaryType
      );
      
      // Save multimodal analysis to database
      const result = await storage.createScan({
        userId: 1,
        ...scanData
      });
      
      res.json(result);
    } catch (error) {
      console.error('Multimodal analysis error:', error);
      res.status(500).json({ message: 'Multimodal analysis failed' });
    }
  });

  // Save user preferences
  app.post('/api/settings/preferences', async (req, res) => {
    try {
      // Validate incoming data
      const preferencesSchema = z.object({
        theme: z.string().optional(),
        language: z.string().optional(),
        confidenceThreshold: z.number().min(50).max(100).optional(),
        enableNotifications: z.boolean().optional(),
        autoAnalyze: z.boolean().optional(),
        sensitivityLevel: z.enum(['low', 'medium', 'high']).optional()
      });
      
      const validatedData = preferencesSchema.parse(req.body);
      
      // In a real app with actual users, we would save to the database here
      // For now, just return success
      res.json({ 
        success: true,
        message: "Preferences saved successfully",
        data: validatedData
      });
    } catch (error) {
      res.status(500).json({ message: 'Failed to save preferences' });
    }
  });

  // Save user profile
  app.post('/api/settings/profile', async (req, res) => {
    try {
      // Validate profile data
      const profileSchema = z.object({
        email: z.string().email().optional(),
        name: z.string().min(2).optional()
      });
      
      const validatedData = profileSchema.parse(req.body);
      
      // In a real app with actual users, we would update the user record here
      // For now, just return success
      res.json({ 
        success: true,
        message: "Profile updated successfully",
        data: validatedData
      });
    } catch (error) {
      res.status(500).json({ message: 'Failed to save profile' });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
