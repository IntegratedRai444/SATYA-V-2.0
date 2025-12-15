import dotenv from 'dotenv';

dotenv.config();

export const pythonConfig = {
  // Base URL of the Python service
  apiUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:8000',
  
  // API Key for authenticating with the Python service
  apiKey: process.env.PYTHON_SERVICE_API_KEY || 'your-secret-api-key',
  
  // Timeout for API requests (in milliseconds)
  requestTimeout: parseInt(process.env.PYTHON_SERVICE_TIMEOUT || '60000', 10),
  
  // Maximum retry attempts for failed requests
  maxRetries: parseInt(process.env.PYTHON_SERVICE_MAX_RETRIES || '3', 10),
  
  // Retry delay (in milliseconds)
  retryDelay: parseInt(process.env.PYTHON_SERVICE_RETRY_DELAY || '1000', 10),
  
  // Enable/disable Python service (for maintenance or debugging)
  enabled: process.env.PYTHON_SERVICE_ENABLED !== 'false',
  
  // Log level for Python service client
  logLevel: process.env.PYTHON_SERVICE_LOG_LEVEL || 'info',
  
  // File upload settings
  upload: {
    // Maximum file size in bytes (default: 50MB)
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '52428800', 10),
    
    // Allowed file types
    allowedTypes: [
      // Images
      'image/jpeg',
      'image/png',
      'image/gif',
      'image/webp',
      
      // Videos
      'video/mp4',
      'video/webm',
      'video/quicktime',
      
      // Audio
      'audio/mpeg',
      'audio/wav',
      'audio/ogg',
      'audio/webm',
    ],
    
    // Temporary directory for file uploads
    tempDir: process.env.UPLOAD_TEMP_DIR || 'uploads',
  },
  
  // Model configurations
  models: {
    // Image analysis model
    image: {
      name: process.env.IMAGE_MODEL_NAME || 'efficientnet_b4',
      version: process.env.IMAGE_MODEL_VERSION || '1.0.0',
      minConfidence: parseFloat(process.env.IMAGE_MIN_CONFIDENCE || '0.7'),
    },
    
    // Video analysis model
    video: {
      name: process.env.VIDEO_MODEL_NAME || 'x3d_xs',
      version: process.env.VIDEO_MODEL_VERSION || '1.0.0',
      minConfidence: parseFloat(process.env.VIDEO_MIN_CONFIDENCE || '0.65'),
      frameSamplingRate: parseInt(process.env.VIDEO_FRAME_RATE || '2', 10),
    },
    
    // Audio analysis model
    audio: {
      name: process.env.AUDIO_MODEL_NAME || 'wav2vec2_large',
      version: process.env.AUDIO_MODEL_VERSION || '1.0.0',
      minConfidence: parseFloat(process.env.AUDIO_MIN_CONFIDENCE || '0.6'),
    },
  },
  
  // Performance metrics
  metrics: {
    // Enable/disable metrics collection
    enabled: process.env.ENABLE_METRICS !== 'false',
    
    // Metrics collection interval (in milliseconds)
    collectionInterval: parseInt(process.env.METRICS_INTERVAL || '60000', 10),
    
    // Metrics retention period (in days)
    retentionDays: parseInt(process.env.METRICS_RETENTION_DAYS || '30', 10),
  },
};

export default pythonConfig;
