// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';

// Application Configuration
export const APP_CONFIG = {
  // Maximum file size for uploads (100MB)
  MAX_FILE_SIZE: 100 * 1024 * 1024,
  
  // Supported file types for upload
  SUPPORTED_FILE_TYPES: [
    'image/jpeg',
    'image/png',
    'image/gif',
    'video/mp4',
    'video/webm'
  ],
  
  // Analysis statuses
  ANALYSIS_STATUS: {
    IDLE: 'idle',
    UPLOADING: 'uploading',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    ERROR: 'error'
  }
};

// Re-export from lib/config for backward compatibility
export * from './lib/config';

export default {
  API_BASE_URL,
  ...APP_CONFIG
};
