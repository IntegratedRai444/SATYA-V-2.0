// API Configuration
const API_CONFIG = {
  // Base URLs
  API_VERSION: 'v2',
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  AUTH_URL: import.meta.env.VITE_AUTH_API_URL || 'http://localhost:5001',
  
  // Timeouts (in milliseconds)
  TIMEOUT: 30000, // 30 seconds
  UPLOAD_TIMEOUT: 120000, // 2 minutes
  
  // Retry configuration
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000, // 1 second
  MAX_RETRY_DELAY: 30000, // 30 seconds
  
  // Cache settings (in milliseconds)
  DEFAULT_CACHE_TTL: 5 * 60 * 1000, // 5 minutes
  
  // Token refresh threshold (in milliseconds before token expiry)
  TOKEN_REFRESH_THRESHOLD: 5 * 60 * 1000, // 5 minutes
  
  // Default headers
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  
  // CORS settings
  CORS: {
    credentials: 'include',
    mode: 'cors',
  },
  
  // Auth token keys
  TOKEN_KEYS: {
    ACCESS_TOKEN: 'satya_access_token',
    REFRESH_TOKEN: 'satya_refresh_token',
    TOKEN_PREFIX: 'Bearer ',
  },
  
  // Error handling
  ERROR_MESSAGES: {
    NETWORK: 'Network error - please check your connection',
    TIMEOUT: 'Request timed out - please try again',
    SERVER: 'Server error - please try again later',
    UNAUTHORIZED: 'Session expired - please log in again',
    FORBIDDEN: 'You do not have permission to perform this action',
    NOT_FOUND: 'The requested resource was not found',
    VALIDATION: 'Please fix the validation errors and try again',
    UNKNOWN: 'An unknown error occurred',
  },
  
  // Debug mode
  DEBUG: import.meta.env.DEV,
} as const;

export default API_CONFIG;
