import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertScanSchema } from "@shared/schema";
import multer from "multer";
import { deepfakeDetector } from "./services/deepfake-detector";
import { advancedDeepfakeDetector } from "./services/advanced-deepfake-detector";
import { 
  startPythonServer, 
  analyzeImage, 
  analyzeVideo, 
  analyzeAudio, 
  analyzeMultimodal, 
  analyzeWebcam 
} from "./python-bridge";

// Setup multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max file size
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Start Python server for advanced AI processing
  try {
    await startPythonServer();
    console.log('Python AI server started successfully');
  } catch (error) {
    console.error('Failed to start Python AI server:', error);
    console.log('Continuing with basic detection capabilities only');
  }
  
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
      const mediaType = type as 'image' | 'video' | 'audio';
      
      if (mediaType === 'image') {
        detectionResult = await advancedDeepfakeDetector.analyzeImage(file.buffer);
      } else if (mediaType === 'video') {
        detectionResult = await advancedDeepfakeDetector.analyzeVideo(file.buffer);
      } else if (mediaType === 'audio') {
        detectionResult = await advancedDeepfakeDetector.analyzeAudio(file.buffer);
      } else {
        return res.status(400).json({ message: 'Unsupported media type' });
      }
      
      // Save scan to database with user ID 1 (demo user)
      const result = await storage.createScan({
        userId: 1,
        filename: file.originalname,
        type: mediaType,
        result: detectionResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: detectionResult.confidence,
        detectionDetails: detectionResult.key_findings,
        metadata: {
          resolution: mediaType === 'image' ? '1920x1080' : undefined,
          duration: mediaType === 'video' || mediaType === 'audio' ? '00:02:34' : undefined,
          size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`
        }
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
      
      // Save to database
      const result = await storage.createScan({
        userId: 1,
        filename: 'webcam_capture.jpg',
        type: 'image',
        result: detectionResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(detectionResult.confidence),
        detectionDetails: detectionResult.key_findings,
        metadata: {
          resolution: '1280x720',
          timestamp: new Date().toISOString()
        }
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
      
      // Save multimodal analysis to database
      const result = await storage.createScan({
        userId: 1,
        filename: filename || 'multimodal_analysis.json',
        type: primaryType,
        result: detectionResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(detectionResult.confidence),
        detectionDetails: detectionResult.key_findings,
        metadata: {
          description: 'Advanced multimodal analysis using cross-modal detection',
          timestamp: new Date().toISOString(),
          combinedFileTypes: Object.keys(files).join('+')
        }
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
  
  // Advanced AI endpoints using Python backend
  
  // Advanced AI image analysis
  app.post('/api/ai/analyze/image', upload.single('image'), async (req, res) => {
    try {
      let imageData;
      
      // Handle image data from request body (webcam) or file upload
      if (req.body.imageData) {
        imageData = req.body.imageData;
      } else if (req.file) {
        // Convert buffer to base64
        imageData = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;
      } else {
        return res.status(400).json({ message: 'No image data provided' });
      }
      
      // Use Python backend for advanced analysis
      const result = await analyzeImage(imageData);
      
      // Save to database
      const scanRecord = await storage.createScan({
        userId: 1, // Demo user
        filename: req.file?.originalname || 'ai_image_scan.jpg',
        type: 'image',
        result: result.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(result.confidence),
        detectionDetails: result.key_findings,
        metadata: {
          analysis_date: result.analysis_date,
          case_id: result.case_id,
          ai_powered: true,
          processor: 'python-ai'
        }
      });
      
      // Return combined result
      res.json({
        ...result,
        id: scanRecord.id,
        saved: true
      });
    } catch (error) {
      console.error('AI image analysis error:', error);
      res.status(500).json({ message: 'AI image analysis failed' });
    }
  });
  
  // Advanced AI video analysis
  app.post('/api/ai/analyze/video', upload.single('video'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: 'No video file provided' });
      }
      
      // Use Python backend for advanced video analysis
      const result = await analyzeVideo(req.file.buffer, req.file.originalname);
      
      // Save to database
      const scanRecord = await storage.createScan({
        userId: 1, // Demo user
        filename: req.file.originalname,
        type: 'video',
        result: result.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(result.confidence),
        detectionDetails: result.key_findings,
        metadata: {
          analysis_date: result.analysis_date,
          case_id: result.case_id,
          ai_powered: true,
          processor: 'python-ai',
          filesize: `${(req.file.size / (1024 * 1024)).toFixed(2)} MB`
        }
      });
      
      // Return combined result
      res.json({
        ...result,
        id: scanRecord.id,
        saved: true
      });
    } catch (error) {
      console.error('AI video analysis error:', error);
      res.status(500).json({ message: 'AI video analysis failed' });
    }
  });
  
  // Advanced AI multimodal analysis
  app.post('/api/ai/analyze/multimodal', upload.fields([
    { name: 'image', maxCount: 1 },
    { name: 'video', maxCount: 1 },
    { name: 'audio', maxCount: 1 }
  ]), async (req, res) => {
    try {
      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      
      if (!files || Object.keys(files).length === 0) {
        return res.status(400).json({ message: 'No files provided for multimodal analysis' });
      }
      
      // Get file buffers and filenames
      const imageBuffer = files.image?.[0]?.buffer;
      const videoBuffer = files.video?.[0]?.buffer;
      const audioBuffer = files.audio?.[0]?.buffer;
      
      // Use Python backend for multimodal analysis
      const result = await analyzeMultimodal(imageBuffer, audioBuffer, videoBuffer);
      
      // Determine primary file type and filename for database record
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
      
      // Save to database
      const scanRecord = await storage.createScan({
        userId: 1, // Demo user
        filename: filename,
        type: primaryType,
        result: result.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(result.confidence),
        detectionDetails: result.key_findings,
        metadata: {
          analysis_date: result.analysis_date,
          case_id: result.case_id,
          ai_powered: true,
          processor: 'python-ai-multimodal',
          modalities_used: result.modalities_used || Object.keys(files).length
        }
      });
      
      // Return combined result
      res.json({
        ...result,
        id: scanRecord.id,
        saved: true
      });
    } catch (error) {
      console.error('AI multimodal analysis error:', error);
      res.status(500).json({ message: 'AI multimodal analysis failed' });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
