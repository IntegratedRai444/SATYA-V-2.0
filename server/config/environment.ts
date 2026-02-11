// Load environment variables first
import * as dotenv from 'dotenv';
import * as path from 'path';

// Configure dotenv to load from server directory (handle both dev and prod)
const envPath = path.resolve(process.cwd(), '.env');
console.log('Loading .env from:', envPath);
dotenv.config({ path: envPath });

// This must be first import to ensure environment variables are loaded
import '../setup-env';

import { z } from 'zod';
import * as fs from 'fs';

// Helper function to get environment variable with fallback
function getEnvVar(key: string, defaultValue?: string): string | undefined {
  return process.env[key] || process.env[`VITE_${key}`] || defaultValue;
}

// Process environment variables with VITE_ prefix fallback
const processedEnv = {
  ...process.env,
  SUPABASE_URL: getEnvVar('SUPABASE_URL'),
  SUPABASE_ANON_KEY: getEnvVar('SUPABASE_ANON_KEY'),
  SUPABASE_SERVICE_ROLE_KEY: getEnvVar('SUPABASE_SERVICE_ROLE_KEY'),
  SUPABASE_JWT_SECRET: getEnvVar('SUPABASE_JWT_SECRET'),
  JWT_SECRET: getEnvVar('JWT_SECRET'),
  JWT_SECRET_KEY: getEnvVar('JWT_SECRET_KEY'),
  SESSION_SECRET: getEnvVar('SESSION_SECRET'),
  FLASK_SECRET_KEY: getEnvVar('FLASK_SECRET_KEY'),
  DATABASE_URL: getEnvVar('DATABASE_URL'),
  DATABASE_PATH: getEnvVar('DATABASE_PATH'),
  UPLOAD_DIR: getEnvVar('UPLOAD_DIR'),
  MAX_FILE_SIZE: getEnvVar('MAX_FILE_SIZE'),
  CLEANUP_INTERVAL: getEnvVar('CLEANUP_INTERVAL'),
  CORS_ORIGIN: getEnvVar('CORS_ORIGIN'),
  RATE_LIMIT_WINDOW_MS: getEnvVar('RATE_LIMIT_WINDOW_MS'),
  RATE_LIMIT_MAX_REQUESTS: getEnvVar('RATE_LIMIT_MAX_REQUESTS'),
  RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS: getEnvVar('RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS'),
  LOG_LEVEL: getEnvVar('LOG_LEVEL'),
  LOG_FORMAT: getEnvVar('LOG_FORMAT'),
  LOG_FILE: getEnvVar('LOG_FILE'),
  HEALTH_CHECK_INTERVAL: getEnvVar('HEALTH_CHECK_INTERVAL'),
  PYTHON_HEALTH_CHECK_URL: getEnvVar('PYTHON_HEALTH_CHECK_URL'),
  ENABLE_METRICS: getEnvVar('ENABLE_METRICS'),
  METRICS_PORT: getEnvVar('METRICS_PORT'),
  ENABLE_DEV_TOOLS: getEnvVar('ENABLE_DEV_TOOLS'),
  HOT_RELOAD: getEnvVar('HOT_RELOAD'),
  PYTHON_SERVER_URL: getEnvVar('PYTHON_SERVER_URL'),
  PYTHON_SERVER_PORT: getEnvVar('PYTHON_SERVER_PORT'),
  PYTHON_SERVER_TIMEOUT: getEnvVar('PYTHON_SERVER_TIMEOUT'),
};

// Environment schema with validation
const envSchema = z.object({
  // Server Configuration
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(5001),
  
  // Supabase Configuration
  SUPABASE_URL: z.string().url('SUPABASE_URL must be a valid URL'),
  SUPABASE_ANON_KEY: z.string().min(10, 'SUPABASE_ANON_KEY must be at least 10 characters'),
  SUPABASE_SERVICE_ROLE_KEY: z.string().min(32, 'SUPABASE_SERVICE_ROLE_KEY must be at least 32 characters'),
  SUPABASE_JWT_SECRET: z.string().min(32, 'SUPABASE_JWT_SECRET must be at least 32 characters'),
  
  // Security Configuration
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters')
    .refine(val => !val.includes('your-secret'), {
      message: 'JWT_SECRET must be changed from default value'
    }),
  JWT_SECRET_KEY: z.string().min(32, 'JWT_SECRET_KEY must be at least 32 characters')
    .refine(val => !val.includes('your-secret'), {
      message: 'JWT_SECRET_KEY must be changed from default value'
    }),
  JWT_EXPIRES_IN: z.string().default('15m'),
  REFRESH_TOKEN_EXPIRES_IN: z.string().default('7d'),
  SESSION_SECRET: z.string().min(32, 'SESSION_SECRET must be at least 32 characters')
    .refine(val => !val.includes('your-secret'), {
      message: 'SESSION_SECRET must be changed from default value'
    }).optional(),
  
  // Python AI Engine Configuration
  PYTHON_SERVER_URL: z.string().url().default('http://localhost:8000'),
  PYTHON_SERVER_PORT: z.coerce.number().default(8000),
  PYTHON_SERVER_TIMEOUT: z.coerce.number().default(300000), // 5 minutes
  FLASK_SECRET_KEY: z.string().min(32, 'FLASK_SECRET_KEY must be at least 32 characters').optional(),
  
  // Database Configuration
  DATABASE_URL: z.string().default('sqlite:./satyaai.db'),
  DATABASE_PATH: z.string().default('./satyaai.db'),
  
  // File Upload Configuration
  UPLOAD_DIR: z.string().default('./uploads'),
  MAX_FILE_SIZE: z.coerce.number().default(104857600), // 100MB
  CLEANUP_INTERVAL: z.coerce.number().default(3600000), // 1 hour
  
  // CORS Configuration - Comma-separated list of allowed origins
  CORS_ORIGIN: z.string().default(''),
  
  // Rate Limiting Configuration
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(900000), // 15 minutes
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(50),
  RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS: z.coerce.boolean().default(true),
  
  // Logging Configuration
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  LOG_FORMAT: z.enum(['json', 'simple']).default('simple'),
  LOG_FILE: z.string().optional(),
  
  // Health Check Configuration
  HEALTH_CHECK_INTERVAL: z.coerce.number().default(30000), // 30 seconds
  PYTHON_HEALTH_CHECK_URL: z.string().url().default('http://localhost:8000/health'),
  
  // Performance Configuration
  ENABLE_METRICS: z.coerce.boolean().default(true),
  METRICS_PORT: z.coerce.number().optional(),
  
  // Redis Configuration
  REDIS_HOST: z.string().default('localhost'),
  REDIS_PORT: z.coerce.number().default(6379),
  REDIS_PASSWORD: z.string().optional(),
  
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
    // For development, require actual environment variables but provide sensible defaults
    if (process.env.NODE_ENV === 'development') {
      // Validate that critical environment variables are present
      const requiredVars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'JWT_SECRET'];
      const missingVars = requiredVars.filter(varName => !process.env[varName]);
      
      if (missingVars.length > 0) {
        throw new ConfigurationError(
          `Missing required environment variables: ${missingVars.join(', ')}`
        );
      }

      // Production safety checks - no fallback secrets allowed
      const weakSecrets = [
        'jwt-secret-key-set-in-env',
        'session-secret-key-set-in-env',
        'csrf-secret-key-set-in-env',
        'your-secret-key-here',
        'change-me-in-production',
        'default-secret-key'
      ];
      
      const secrets = ['JWT_SECRET', 'SESSION_SECRET', 'CSRF_SECRET'];
      for (const secret of secrets) {
        const value = process.env[secret];
        if (value && weakSecrets.some(weak => value.toLowerCase().includes(weak.toLowerCase()))) {
          throw new ConfigurationError(
            `Weak or default secret detected for ${secret}. Please use a cryptographically secure secret.`
          );
        }
      }

      return {
        NODE_ENV: 'development' as const,
        PORT: parseInt(process.env.PORT || '5001'),
        SUPABASE_URL: process.env.SUPABASE_URL!,
        SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY!,
        SUPABASE_SERVICE_ROLE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY!,
        SUPABASE_JWT_SECRET: process.env.SUPABASE_JWT_SECRET || process.env.JWT_SECRET!,
        JWT_SECRET: process.env.JWT_SECRET!,
        JWT_SECRET_KEY: process.env.JWT_SECRET_KEY || process.env.JWT_SECRET!,
        SESSION_SECRET: process.env.SESSION_SECRET || process.env.JWT_SECRET!,
        FLASK_SECRET_KEY: process.env.FLASK_SECRET_KEY || process.env.JWT_SECRET!,
        DATABASE_URL: process.env.DATABASE_URL || 'sqlite:./satyaai.db',
        DATABASE_PATH: process.env.DATABASE_PATH || './satyaai.db',
        UPLOAD_DIR: process.env.UPLOAD_DIR || './uploads',
        MAX_FILE_SIZE: parseInt(process.env.MAX_FILE_SIZE || '104857600'),
        CLEANUP_INTERVAL: parseInt(process.env.CLEANUP_INTERVAL || '3600000'),
        CORS_ORIGIN: process.env.CORS_ORIGIN || 'http://localhost:5173,http://localhost:3000,http://localhost:5001',
        RATE_LIMIT_WINDOW_MS: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000'),
        RATE_LIMIT_MAX_REQUESTS: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100'),
        RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS: process.env.RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS === 'true',
        LOG_LEVEL: (process.env.LOG_LEVEL as 'error' | 'warn' | 'info' | 'debug') || 'debug',
        LOG_FORMAT: (process.env.LOG_FORMAT as 'json' | 'simple') || 'simple',
        LOG_FILE: process.env.LOG_FILE || 'logs/app.log',
        HEALTH_CHECK_INTERVAL: parseInt(process.env.HEALTH_CHECK_INTERVAL || '30000'),
        PYTHON_HEALTH_CHECK_URL: process.env.PYTHON_HEALTH_CHECK_URL || 'http://localhost:8000/health',
        ENABLE_METRICS: process.env.ENABLE_METRICS !== 'false',
        ENABLE_DEV_TOOLS: process.env.ENABLE_DEV_TOOLS !== 'false',
        HOT_RELOAD: process.env.HOT_RELOAD !== 'false',
        PYTHON_SERVER_URL: process.env.PYTHON_SERVER_URL || 'http://localhost:8000',
        PYTHON_SERVER_PORT: parseInt(process.env.PYTHON_SERVER_PORT || '8000'),
        PYTHON_SERVER_TIMEOUT: parseInt(process.env.PYTHON_SERVER_TIMEOUT || '300000'),
        JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '15m',
        REFRESH_TOKEN_EXPIRES_IN: process.env.REFRESH_TOKEN_EXPIRES_IN || '7d',
        REDIS_HOST: process.env.REDIS_HOST || 'localhost',
        REDIS_PORT: parseInt(process.env.REDIS_PORT || '6379'),
        REDIS_PASSWORD: process.env.REDIS_PASSWORD,
      } as Environment;
    }
    
    // Parse and validate environment variables
    const parsed = envSchema.parse(processedEnv);
    
    // Always validate security configuration
    validateProductionConfig(parsed);
    
    // Ensure upload directory exists
    ensureDirectoryExists(parsed.UPLOAD_DIR);
    
    return parsed;
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Handle Zod validation errors
      const errorMessages = error.issues.map(issue => 
        `${issue.path.join('.')}: ${issue.message}`
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
  const securityChecks = [
    {
      condition: config.NODE_ENV === 'production' && config.JWT_SECRET.length < 32,
      message: 'JWT_SECRET must be at least 32 characters long in production'
    },
    {
      condition: config.NODE_ENV === 'production' && config.SESSION_SECRET && config.SESSION_SECRET.length < 32,
      message: 'SESSION_SECRET must be at least 32 characters long in production'
    },
    {
      condition: config.NODE_ENV === 'production' && config.FLASK_SECRET_KEY && config.FLASK_SECRET_KEY.length < 32,
      message: 'FLASK_SECRET_KEY must be at least 32 characters long in production'
    },
    {
      condition: config.NODE_ENV === 'production' && config.CORS_ORIGIN.includes('*'),
      message: 'CORS_ORIGIN should not use wildcard (*) in production'
    },
    {
      condition: config.NODE_ENV === 'production' && (!config.JWT_EXPIRES_IN || config.JWT_EXPIRES_IN === '1h'),
      message: 'JWT_EXPIRES_IN should be set to a secure value (e.g., 15m) in production'
    }
  ];

  const failures = securityChecks.filter(check => check.condition);
  
  if (failures.length > 0) {
    const envPrefix = config.NODE_ENV === 'production' ? 'Production' : 'Development';
    const messages = failures.map(f => `[${envPrefix}] ${f.message}`).join('\n');
    if (config.NODE_ENV === 'production') {
      throw new ConfigurationError(`Security configuration issues:\n${messages}`);
    }
    console.warn(`Security warnings (${config.NODE_ENV}):\n${messages}`);
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