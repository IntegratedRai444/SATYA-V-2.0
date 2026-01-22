import { Router, Request, Response } from 'express';
import { supabaseAuth } from '../middleware/supabase-auth';
import { logger } from '../config/logger';

const router = Router();

// Model information (static for now, can be enhanced to fetch from Python backend)
const availableModels = [
  {
    id: 'efficientnet-b4',
    name: 'EfficientNet-B4',
    version: '1.0.0',
    description: 'EfficientNet-B4 model for image deepfake detection',
    isActive: true,
    supportedTypes: ['image'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'xception',
    name: 'Xception',
    version: '1.0.0',
    description: 'Xception model for image deepfake detection',
    isActive: true,
    supportedTypes: ['image'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'video-detector',
    name: 'Video Deepfake Detector',
    version: '1.0.0',
    description: 'CNN-based model for video deepfake detection',
    isActive: true,
    supportedTypes: ['video'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'audio-detector',
    name: 'Audio Deepfake Detector',
    version: '1.0.0',
    description: 'Spectrogram-based model for audio deepfake detection',
    isActive: true,
    supportedTypes: ['audio'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'multimodal-fusion',
    name: 'Multimodal Fusion Detector',
    version: '1.0.0',
    description: 'Fusion model for multimodal deepfake detection',
    isActive: true,
    supportedTypes: ['image', 'video', 'audio', 'multimodal'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }
];

// GET /api/v2/models - Get available analysis models
router.get('/', supabaseAuth, async (req: Request, res: Response) => {
  try {
    const { type, activeOnly, limit = 20, offset = 0 } = req.query;

    let filteredModels = availableModels;

    // Filter by type if specified
    if (type && typeof type === 'string') {
      filteredModels = filteredModels.filter(model => 
        model.supportedTypes.includes(type as 'image' | 'video' | 'audio' | 'multimodal')
      );
    }

    // Filter active models only if specified
    if (activeOnly === 'true') {
      filteredModels = filteredModels.filter(model => model.isActive);
    }

    // Apply pagination
    const startIndex = Number(offset);
    const endIndex = startIndex + Number(limit);
    const paginatedModels = filteredModels.slice(startIndex, endIndex);

    res.json({
      success: true,
      data: {
        items: paginatedModels,
        total: filteredModels.length,
        limit: Number(limit),
        offset: Number(offset)
      }
    });
  } catch (error) {
    logger.error('Error fetching models:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

// GET /api/v2/models/:id - Get specific model information
router.get('/:id', supabaseAuth, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    
    const model = availableModels.find(m => m.id === id);
    
    if (!model) {
      return res.status(404).json({
        success: false,
        error: 'Model not found'
      });
    }

    res.json({
      success: true,
      data: model
    });
  } catch (error) {
    logger.error('Error fetching model:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

export { router as modelsRouter };
