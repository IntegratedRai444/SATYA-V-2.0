// Feature flags configuration
export const features = {
  // Enable/disable metrics endpoint
  enableMetrics: process.env.ENABLE_METRICS === 'true' || process.env.NODE_ENV === 'production',
  
  // Enable/disable WebSocket functionality
  enableWebSockets: process.env.ENABLE_WEBSOCKETS === 'true' || false,
  
  // Enable/disable Python service integration
  enablePythonService: process.env.ENABLE_PYTHON_SERVICE === 'true' || true,
  
  // Enable/disable API documentation
  enableApiDocs: process.env.NODE_ENV !== 'production',
  
  // Enable/disable rate limiting
  enableRateLimiting: process.env.ENABLE_RATE_LIMITING !== 'false',
  
  // Enable/disable request logging
  enableRequestLogging: process.env.ENABLE_REQUEST_LOGGING !== 'false',
  
  // Enable/disable database query logging
  enableQueryLogging: process.env.ENABLE_QUERY_LOGGING === 'true',
  
  // Enable/disable file uploads
  enableFileUploads: process.env.ENABLE_FILE_UPLOADS !== 'false',
  
  // Enable/disable user authentication
  enableAuth: process.env.ENABLE_AUTH !== 'false',
  
  // Development mode features
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
  
  // Get a feature flag value
  get: (key: string, defaultValue: any = false) => {
    return key in features ? features[key as keyof typeof features] : defaultValue;
  }
} as const;

export type FeatureFlags = typeof features;
export default features;
