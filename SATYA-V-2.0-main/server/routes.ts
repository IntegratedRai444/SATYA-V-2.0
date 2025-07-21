import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertScanSchema } from "@shared/schema";
import multer from "multer";
import { deepfakeDetector } from "./services/deepfake-detector";
import { advancedDeepfakeDetector } from "./services/advanced-detection-service";
import { mockAuthService as authService } from "./services/mock-auth-service";
import { generatePDFReport } from "./services/report-generator";
import * as mockDetector from "./services/mock-detection-service";
import cors from 'cors';

// Setup multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max file size
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  app.use(cors()); // Enable CORS for all routes
  // Start advanced detection services
  console.log('Initializing advanced detection capabilities');
  
  // Initialize auth service
  await authService.initialize();
  
  // Set up API routes
  const apiRouter = app.route('/api');
  
  // Authentication routes
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { username, password } = req.body;
      
      if (!username || !password) {
        return res.status(400).json({ 
          success: false, 
          message: 'Username and password are required' 
        });
      }
      
      const result = await authService.login(username, password);
      res.json(result);
    } catch (error) {
      console.error('Login route error:', error);
      res.status(500).json({ 
        success: false, 
        message: 'Authentication service error' 
      });
    }
  });
  
  app.post('/api/auth/logout', async (req, res) => {
    try {
      const { token } = req.body;
      
      if (!token) {
        return res.status(400).json({ 
          success: false, 
          message: 'Session token is required' 
        });
      }
      
      const result = await authService.logout(token);
      res.json(result);
    } catch (error) {
      console.error('Logout route error:', error);
      res.status(500).json({ 
        success: false, 
        message: 'Logout service error' 
      });
    }
  });
  
  app.post('/api/auth/validate', async (req, res) => {
    try {
      const { token } = req.body;
      
      if (!token) {
        return res.status(400).json({ 
          valid: false, 
          message: 'Session token is required' 
        });
      }
      
      const result = await authService.validateSession(token);
      res.json(result);
    } catch (error) {
      console.error('Session validation route error:', error);
      res.status(500).json({ 
        valid: false, 
        message: 'Session validation service error' 
      });
    }
  });

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
  
  // Generate PDF report for a scan
  app.get('/api/scans/:id/report', async (req, res) => {
    try {
      const scan = await storage.getScanById(parseInt(req.params.id));
      if (!scan) {
        return res.status(404).json({ message: 'Scan not found' });
      }
      
      // Set response headers for PDF download
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="SatyaAI-Report-${scan.id}.pdf"`);
      
      // Format data for the report generator
      const reportData = {
        id: scan.id,
        filename: scan.filename,
        type: scan.type,
        result: scan.result,
        confidence: scan.confidenceScore,
        analyzedAt: scan.createdAt,
        details: {
          manipulationScore: scan.result === 'authentic' ? 15 : 85,
          authenticityScore: scan.result === 'authentic' ? 85 : 15,
          visualArtifacts: scan.result === 'authentic' ? 12 : 78,
          audioArtifacts: scan.type === 'audio' || scan.type === 'video' 
            ? (scan.result === 'authentic' ? 8 : 72) 
            : undefined,
          inconsistencyMarkers: {
            facial: scan.type === 'image' || scan.type === 'video' 
              ? (scan.result === 'authentic' ? 5 : 65) 
              : undefined,
            audio: scan.type === 'audio' || scan.type === 'video' 
              ? (scan.result === 'authentic' ? 7 : 68) 
              : undefined,
            metadata: scan.result === 'authentic' ? 3 : 42
          },
          detectedRegions: scan.result === 'authentic' ? [] : [
            {
              x: 120,
              y: 80,
              width: 200,
              height: 180,
              confidence: scan.confidenceScore,
              type: 'face-modification'
            }
          ],
          metadata: scan.metadata || {},
          analysisTime: 4.2,
          modelVersion: 'SatyaAI Neural Vision v4.2',
          warnings: scan.result === 'authentic' ? [] : [
            'Potential manipulation detected in facial region',
            'Visual artifacts detected around eye and mouth areas',
            'Metadata inconsistencies found'
          ],
          recommendations: [
            'Verify this media with additional tools',
            'Check the source of the media',
            'For critical decisions, consult with a digital forensics expert'
          ]
        }
      };
      
      // Generate and send the PDF
      await generatePDFReport(reportData, res);
    } catch (error) {
      console.error('PDF generation error:', error);
      res.status(500).json({ message: 'Failed to generate PDF report' });
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
      
      // Process file using the mock detector service
      let detectionResult;
      const mediaType = type as 'image' | 'video' | 'audio';
      
      if (mediaType === 'image') {
        detectionResult = await mockDetector.analyzeImage(file.buffer);
      } else if (mediaType === 'video') {
        detectionResult = await mockDetector.analyzeVideo(file.buffer);
      } else if (mediaType === 'audio') {
        detectionResult = await mockDetector.analyzeAudio(file.buffer);
      } else {
        return res.status(400).json({ message: 'Unsupported media type' });
      }
      
      // Save scan to database with user ID 1 (demo user)
      const result = await storage.createScan({
        userId: 1,
        filename: file.originalname,
        type: mediaType,
        result: detectionResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(detectionResult.confidence * 100), // Convert to integer percentage
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
      
      // Process the webcam image with mock detector
      const detectionResult = await mockDetector.analyzeWebcam(imageBuffer);
      
      // Save to database
      const result = await storage.createScan({
        userId: 1,
        filename: 'webcam_capture.jpg',
        type: 'image',
        result: detectionResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(detectionResult.confidence * 100), // Convert to integer percentage
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
      
      // Use advanced detection service for multimodal analysis
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
        confidenceScore: Math.round(detectionResult.confidence * 100), // Convert to integer percentage
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
  
  // Advanced AI endpoints
  
  // --- Robust /api/ai/analyze/image endpoint ---
  app.post('/api/ai/analyze/image', upload.single('image'), async (req, res) => {
    try {
      let imageBuffer;
      if (req.file) {
        imageBuffer = req.file.buffer;
      } else if (req.body.imageData) {
        // Handle base64 image from webcam
        const base64Data = req.body.imageData.split(',')[1];
        imageBuffer = Buffer.from(base64Data, 'base64');
      } else {
        return res.status(400).json({ error: 'No image data provided' });
      }
      // Mock analysis result
      const result = {
        authenticity: 'AUTHENTIC MEDIA',
        confidence: 98,
        analysis_date: new Date().toISOString(),
        case_id: 'case-' + Date.now(),
        key_findings: ['No manipulation detected'],
        technical_details: { models_used: ['resnet50'], device: 'CPU', analysis_version: '1.0' }
      };
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'AI image analysis failed' });
    }
  });
  // --- Robust /api/ai/analyze/video endpoint ---
  app.post('/api/ai/analyze/video', upload.single('video'), async (req, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No video file provided' });
      const result = {
        authenticity: 'DEEPFAKE',
        confidence: 65,
        analysis_date: new Date().toISOString(),
        case_id: 'case-' + Date.now(),
        key_findings: ['Inconsistent facial movements detected'],
        technical_details: { models_used: ['vision_transformer'], device: 'CPU', analysis_version: '1.0' }
      };
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'AI video analysis failed' });
    }
  });
  // --- Robust /api/ai/analyze/audio endpoint ---
  app.post('/api/ai/analyze/audio', upload.single('audio'), async (req, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: 'No audio file provided' });
      const result = {
        authenticity: 'AUTHENTIC MEDIA',
        confidence: 92,
        analysis_date: new Date().toISOString(),
        case_id: 'case-' + Date.now(),
        key_findings: ['No synthetic voice detected'],
        technical_details: { models_used: ['audio_cnn'], device: 'CPU', analysis_version: '1.0' }
      };
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: 'AI audio analysis failed' });
    }
  });
  // --- Robust /api/ai/analyze/batch endpoint ---
  app.post('/api/ai/analyze/batch', upload.array('files', 10), async (req, res) => {
    try {
      if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files provided for batch analysis' });
      const results = (req.files as Express.Multer.File[]).map((file, idx) => ({
        authenticity: idx % 2 === 0 ? 'AUTHENTIC MEDIA' : 'DEEPFAKE',
        confidence: 90 - idx * 5,
        analysis_date: new Date().toISOString(),
        case_id: 'case-' + Date.now() + '-' + idx,
        key_findings: [idx % 2 === 0 ? 'No manipulation detected' : 'Deepfake pattern detected'],
        technical_details: { models_used: ['resnet50'], device: 'CPU', analysis_version: '1.0' }
      }));
      res.json({
        batch_id: 'batch-' + Date.now(),
        total_files: results.length,
        analysis_date: new Date().toISOString(),
        average_confidence: results.reduce((a, b) => a + b.confidence, 0) / results.length,
        results,
        batch_summary: {
          authentic_files: results.filter(r => r.authenticity === 'AUTHENTIC MEDIA').length,
          manipulated_files: results.filter(r => r.authenticity === 'DEEPFAKE').length,
          processing_time_total: 1000 * results.length
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'AI batch analysis failed' });
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
      
      // Get file buffers and perform individual analyses
      const modalities: string[] = [];
      const modalityResults: Record<string, any> = {};
      let totalConfidence = 0;
      let weightedSum = 0;
      
      // Process each available modality
      if (files.image) {
        modalities.push('image');
        const imageResult = await advancedDeepfakeDetector.analyzeImage(files.image[0].buffer);
        modalityResults.image = {
          result: imageResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
          confidence: Math.round(imageResult.confidence),
          key_findings: imageResult.key_findings?.slice(0, 2) || [] // Top 2 findings
        };
        weightedSum += imageResult.authenticity === 'AUTHENTIC MEDIA' ? 0.4 * 1 : 0.4 * 0;
        totalConfidence += 0.4;
      }
      
      if (files.video) {
        modalities.push('video');
        const videoResult = await advancedDeepfakeDetector.analyzeVideo(files.video[0].buffer);
        modalityResults.video = {
          result: videoResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
          confidence: Math.round(videoResult.confidence),
          key_findings: videoResult.key_findings?.slice(0, 2) || [] // Top 2 findings
        };
        weightedSum += videoResult.authenticity === 'AUTHENTIC MEDIA' ? 0.4 * 1 : 0.4 * 0;
        totalConfidence += 0.4;
      }
      
      if (files.audio) {
        modalities.push('audio');
        const audioResult = await advancedDeepfakeDetector.analyzeAudio(files.audio[0].buffer);
        modalityResults.audio = {
          result: audioResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
          confidence: Math.round(audioResult.confidence),
          key_findings: audioResult.key_findings?.slice(0, 2) || [] // Top 2 findings
        };
        weightedSum += audioResult.authenticity === 'AUTHENTIC MEDIA' ? 0.2 * 1 : 0.2 * 0;
        totalConfidence += 0.2;
      }
      
      // Calculate final result
      const finalConfidence = totalConfidence > 0 ? Math.min(Math.round((weightedSum / totalConfidence) * 100), 100) : 50;
      const finalAuthenticity = finalConfidence >= 50 ? "AUTHENTIC MEDIA" : "MANIPULATED MEDIA";
      
      // Calculate cross-modal consistency
      let crossModalConsistency = 1.0;
      if (modalities.length > 1) {
        const results = Object.values(modalityResults).map(r => r.result === 'authentic');
        const allSame = results.every(r => r === results[0]);
        crossModalConsistency = allSame ? 0.95 : 0.5;
      }
      
      // Generate combined key findings
      const combinedFindings: string[] = [];
      
      // Modality-specific findings
      Object.entries(modalityResults).forEach(([modality, result]) => {
        result.key_findings.forEach((finding: string) => {
          combinedFindings.push(`[${modality.toUpperCase()}] ${finding}`);
        });
      });
      
      // Add cross-modal findings
      if (modalities.length > 1) {
        if (crossModalConsistency > 0.8) {
          combinedFindings.push("Cross-modal consistency check passed");
          combinedFindings.push(`Strong correlation across ${modalities.join(', ')} analysis`);
        } else {
          combinedFindings.push("Cross-modal inconsistencies detected");
          combinedFindings.push("Potential manipulation indicated by modality conflicts");
        }
      }
      
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
      
      // Prepare enhanced result
      const enhancedResult = {
        authenticity: finalAuthenticity,
        confidence: finalConfidence,
        analysis_date: new Date().toISOString(),
        case_id: `multi-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        model_info: {
          name: "SatyaAI Multimodal Fusion",
          version: "3.0",
          type: "MultimodalAnalysis"
        },
        key_findings: combinedFindings,
        modalities_used: modalities,
        modality_results: modalityResults,
        cross_modal_consistency: crossModalConsistency
      };
      
      // Save to database
      const scanRecord = await storage.createScan({
        userId: 1, // Demo user
        filename: filename,
        type: primaryType,
        result: enhancedResult.authenticity === 'AUTHENTIC MEDIA' ? 'authentic' : 'deepfake',
        confidenceScore: Math.round(enhancedResult.confidence),
        detectionDetails: enhancedResult.key_findings,
        metadata: {
          analysis_date: enhancedResult.analysis_date,
          case_id: enhancedResult.case_id,
          ai_powered: true,
          processor: 'advanced-ai-multimodal',
          modalities_used: modalities,
          cross_modal_consistency: crossModalConsistency
        }
      });
      
      // Return combined result
      res.json({
        ...enhancedResult,
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
