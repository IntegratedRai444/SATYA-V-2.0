import { 
  config, 
  getConfig, 
  getDatabaseConfig, 
  getCorsConfig, 
  getRateLimitConfig, 
  getUploadConfig,
  ConfigurationError,
  type Environment 
} from './environment';
import { logger, logError, logSecurity } from './logger';

// Export all configuration
export {
  config,
  getConfig,
  getDatabaseConfig,
  getCorsConfig,
  getRateLimitConfig,
  getUploadConfig,
  ConfigurationError,
  logger,
  logError,
  logSecurity,
  type Environment
};

// Configuration validation and startup checks
export async function validateConfiguration(): Promise<boolean> {
  try {
    logger.info('Validating configuration...');
    
    // Test database path accessibility
    const dbConfig = getDatabaseConfig(config);
    logger.info('Database configuration loaded', { 
      url: dbConfig.url,
      path: dbConfig.path 
    });
    
    // Test upload directory
    const uploadConfig = getUploadConfig(config);
    logger.info('Upload configuration loaded', {
      uploadDir: uploadConfig.uploadDir,
      maxFileSize: `${Math.round(uploadConfig.maxFileSize / 1024 / 1024)}MB`
    });
    
    // Test Python server connectivity (optional at startup)
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(config.PYTHON_SERVER_URL + '/health', {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        logger.info('Python AI engine is accessible', { 
          url: config.PYTHON_SERVER_URL 
        });
      } else {
        logger.warn('Python AI engine returned non-200 status', { 
          url: config.PYTHON_SERVER_URL,
          status: response.status 
        });
      }
    } catch (error) {
      logger.warn('Python AI engine not accessible at startup (this is OK)', { 
        url: config.PYTHON_SERVER_URL,
        error: (error as Error).message 
      });
    }
    
    logger.info('Configuration validation completed successfully');
    return true;
    
  } catch (error) {
    if (error instanceof ConfigurationError) {
      logger.error('Configuration validation failed', {
        message: error.message,
        errors: error.errors?.issues?.map(issue => ({
          path: issue.path.join('.'),
          message: issue.message,
          code: issue.code
        }))
      });
    } else {
      logError(error as Error, { context: 'configuration_validation' });
    }
    return false;
  }
}

// Environment-specific configurations
export const isDevelopment = config.NODE_ENV === 'development';
export const isProduction = config.NODE_ENV === 'production';
export const isTest = config.NODE_ENV === 'test';

// Server configuration
export const serverConfig = {
  port: config.PORT,
  host: '0.0.0.0',
  cors: getCorsConfig(config),
  rateLimit: getRateLimitConfig(config),
  upload: getUploadConfig(config),
  database: getDatabaseConfig(config),
};

// Python AI engine configuration
export const pythonConfig = {
  url: config.PYTHON_SERVER_URL,
  port: config.PYTHON_SERVER_PORT,
  timeout: config.PYTHON_SERVER_TIMEOUT,
  healthCheckUrl: config.PYTHON_HEALTH_CHECK_URL,
  healthCheckInterval: config.HEALTH_CHECK_INTERVAL,
};

// Security configuration
export const securityConfig = {
  jwtSecret: config.JWT_SECRET,
  jwtExpiresIn: config.JWT_EXPIRES_IN,
  sessionSecret: config.SESSION_SECRET,
  corsOrigins: config.CORS_ORIGIN.split(',').map(o => o.trim()),
};

// Feature flags
export const features = {
  enableMetrics: config.ENABLE_METRICS,
  enableDevTools: config.ENABLE_DEV_TOOLS && isDevelopment,
  hotReload: config.HOT_RELOAD && isDevelopment,
};

// Health check configuration
export const healthConfig = {
  interval: config.HEALTH_CHECK_INTERVAL,
  pythonHealthUrl: config.PYTHON_HEALTH_CHECK_URL,
  timeout: 5000,
};

// Logging configuration summary
export function logConfigurationSummary(): void {
  logger.info('SatyaAI Server Configuration Summary', {
    environment: config.NODE_ENV,
    port: config.PORT,
    pythonServer: config.PYTHON_SERVER_URL,
    database: config.DATABASE_URL,
    uploadDir: config.UPLOAD_DIR,
    maxFileSize: `${Math.round(config.MAX_FILE_SIZE / 1024 / 1024)}MB`,
    logLevel: config.LOG_LEVEL,
    corsOrigins: securityConfig.corsOrigins,
    features: {
      metrics: features.enableMetrics,
      devTools: features.enableDevTools,
      hotReload: features.hotReload,
    }
  });
}