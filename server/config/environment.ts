import { z } from 'zod';
import path from 'path';
import fs from 'fs';

// Environment schema with validation
const envSchema = z.object({
  // Server Configuration
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(3000),
  
  // Security Configuration
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters'),
  JWT_SECRET_KEY: z.string().min(32, 'JWT_SECRET_KEY must be at least 32 characters'),
  JWT_EXPIRES_IN: z.string().default('24h'),
  SESSION_SECRET: z.string().min(32, 'SESSION_SECRET must be at least 32 characters'),
  
  // Python AI Engine Configuration
  PYTHON_SERVER_URL: z.string().url().default('http://localhost:5001'),
  PYTHON_SERVER_PORT: z.coerce.number().default(5001),
  PYTHON_SERVER_TIMEOUT: z.coerce.number().default(300000), // 5 minutes
  FLASK_SECRET_KEY: z.string().min(32, 'FLASK_SECRET_KEY must be at least 32 characters'),
  
  // Database Configuration
  DATABASE_URL: z.string().default('sqlite:./satyaai.db'),
  DATABASE_PATH: z.string().default('./satyaai.db'),
  
  // File Upload Configuration
  UPLOAD_DIR: z.string().default('./uploads'),
  MAX_FILE_SIZE: z.coerce.number().default(104857600), // 100MB
  CLEANUP_INTERVAL: z.coerce.number().default(3600000), // 1 hour
  
  // CORS Configuration
  CORS_ORIGIN: z.string().default('http://localhost:3000,http://localhost:5173'),
  
  // Rate Limiting Configuration
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(900000), // 15 minutes
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(100),
  RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS: z.coerce.boolean().default(false),
  
  // Logging Configuration
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  LOG_FORMAT: z.enum(['json', 'simple']).default('simple'),
  LOG_FILE: z.string().optional(),
  
  // Health Check Configuration
  HEALTH_CHECK_INTERVAL: z.coerce.number().default(30000), // 30 seconds
  PYTHON_HEALTH_CHECK_URL: z.string().url().default('http://localhost:5001/health'),
  
  // Performance Configuration
  ENABLE_METRICS: z.coerce.boolean().default(true),
  METRICS_PORT: z.coerce.number().optional(),
  
  // Development Configuration
  ENABLE_DEV_TOOLS: z.coerce.boolean().default(false),
  HOT_RELOAD: z.coerce.boolean().default(true),
});

export type Environment = z.infer<typeof envSchema>;

class ConfigurationError extends Error {
  constructor(message: string, public errors?: z.ZodError) {
    super(message);
    this.name = 'ConfigurationError';
  }
}

/**
 * Load and validate environment configuration
 */
function loadEnvironment(): Environment {
  try {
    // Parse and validate environment variables
    const parsed = envSchema.parse(process.env);
    
    // Additional validation for production environment
    if (parsed.NODE_ENV === 'production') {
      validateProductionConfig(parsed);
    }
    
    // Ensure upload directory exists
    ensureDirectoryExists(parsed.UPLOAD_DIR);
    
    return parsed;
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errorMessages = error.errors.map(err => 
        `${err.path.join('.')}: ${err.message}`
      ).join('\n');
      
      throw new ConfigurationError(
        `Environment configuration validation failed:\n${errorMessages}`,
        error
      );
    }
    throw error;
  }
}

/**
 * Additional validation for production environment
 */
function validateProductionConfig(config: Environment): void {
  const productionChecks = [
    {
      condition: config.JWT_SECRET === 'your-super-secret-jwt-key-that-is-at-least-32-characters-long',
      message: 'JWT_SECRET must be changed from default value in production'
    },
    {
      condition: config.SESSION_SECRET === 'your-super-secret-session-key-that-is-at-least-32-characters-long',
      message: 'SESSION_SECRET must be changed from default value in production'
    },
    {
      condition: config.FLASK_SECRET_KEY === 'your-super-secret-flask-key-that-is-at-least-32-characters-long',
      message: 'FLASK_SECRET_KEY must be changed from default value in production'
    },
    {
      condition: config.CORS_ORIGIN.includes('localhost'),
      message: 'CORS_ORIGIN should not include localhost in production'
    }
  ];

  const failures = productionChecks.filter(check => check.condition);
  
  if (failures.length > 0) {
    const messages = failures.map(f => f.message).join('\n');
    throw new ConfigurationError(`Production configuration issues:\n${messages}`);
  }
}

/**
 * Ensure directory exists, create if it doesn't
 */
function ensureDirectoryExists(dirPath: string): void {
  const resolvedPath = path.resolve(dirPath);
  
  if (!fs.existsSync(resolvedPath)) {
    fs.mkdirSync(resolvedPath, { recursive: true });
  }
}

/**
 * Get configuration for specific environment
 */
export function getConfig(): Environment {
  return loadEnvironment();
}

/**
 * Get database configuration
 */
export function getDatabaseConfig(config: Environment) {
  return {
    url: config.DATABASE_URL,
    path: config.DATABASE_PATH,
    // Add connection pool settings for production
    ...(config.NODE_ENV === 'production' && {
      pool: {
        min: 2,
        max: 10,
        acquireTimeoutMillis: 30000,
        createTimeoutMillis: 30000,
        destroyTimeoutMillis: 5000,
        idleTimeoutMillis: 30000,
        reapIntervalMillis: 1000,
        createRetryIntervalMillis: 100,
      }
    })
  };
}

/**
 * Get CORS configuration
 */
export function getCorsConfig(config: Environment) {
  const origins = config.CORS_ORIGIN.split(',').map(origin => origin.trim());
  
  return {
    origin: config.NODE_ENV === 'development' ? origins : origins.filter(o => !o.includes('localhost')),
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
    maxAge: 86400 // 24 hours
  };
}

/**
 * Get rate limiting configuration
 */
export function getRateLimitConfig(config: Environment) {
  return {
    windowMs: config.RATE_LIMIT_WINDOW_MS,
    max: config.RATE_LIMIT_MAX_REQUESTS,
    skipSuccessfulRequests: config.RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS,
    standardHeaders: true,
    legacyHeaders: false,
    message: {
      error: 'Too many requests',
      message: 'Rate limit exceeded. Please try again later.',
      retryAfter: Math.ceil(config.RATE_LIMIT_WINDOW_MS / 1000)
    }
  };
}

/**
 * Get file upload configuration
 */
export function getUploadConfig(config: Environment) {
  return {
    uploadDir: config.UPLOAD_DIR,
    maxFileSize: config.MAX_FILE_SIZE,
    cleanupInterval: config.CLEANUP_INTERVAL,
    allowedMimeTypes: {
      image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
      video: ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'],
      audio: ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a']
    },
    maxFileSizes: {
      image: 10 * 1024 * 1024, // 10MB
      video: 100 * 1024 * 1024, // 100MB
      audio: 50 * 1024 * 1024 // 50MB
    }
  };
}

// Export the loaded configuration
export const config = getConfig();

// Export configuration error for error handling
export { ConfigurationError };