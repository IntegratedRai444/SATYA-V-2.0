import dotenv from 'dotenv';
import { z } from 'zod';
import { logger } from '../../config/logger';

// Load environment variables from .env file
dotenv.config();

// Type definitions for environment variables
type EnvConfig = {
  // Core Configuration
  NODE_ENV: 'development' | 'production' | 'test';
  PORT: string;
  
  // API Configuration
  VITE_API_URL: string;
  API_BASE_URL: string;
  
  // Security
  JWT_SECRET: string;
  JWT_ALGORITHM: string;
  JWT_EXPIRES_IN: string;
  SESSION_SECRET: string;
  
  // Database
  DATABASE_URL: string;
  
  // Supabase
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  
  // WebSocket
  VITE_WS_URL: string;
  
  // CORS
  CORS_ORIGIN: string;
  
  // Python AI Engine
  PYTHON_SERVER_URL: string;
  PYTHON_SERVER_PORT: string;
  PYTHON_HEALTH_CHECK_URL: string;
  
  // Optional Features
  ENABLE_METRICS?: string;
  ENABLE_DEV_TOOLS?: string;
};

// Environment schema with validation
const envSchema = z.object({
  // Core Configuration
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.string().default('3000'),
  
  // API Configuration
  VITE_API_URL: z.string()
    .url('VITE_API_URL must be a valid URL')
    .refine(url => new URL(url).pathname.endsWith('/api/v2'), {
      message: 'VITE_API_URL must end with /api/v2'
    }),
  API_BASE_URL: z.string()
    .url('API_BASE_URL must be a valid URL')
    .refine(url => !url.endsWith('/'), {
      message: 'API_BASE_URL should not end with a trailing slash'
    }),
  
  // Security
  JWT_SECRET: z.string()
    .min(32, 'JWT_SECRET must be at least 32 characters long')
    .refine(secret => {
      const isDefault = [
        'your-super-secret-jwt-key-that-is-at-least-32-characters-long',
        'change-this-to-a-secure-secret-key',
        'dev-secret-key-change-in-production'
      ].includes(secret);
      return process.env.NODE_ENV !== 'production' || !isDefault;
    }, 'JWT_SECRET must be changed from default value in production'),
    
  JWT_ALGORITHM: z.string().default('HS256'),
  JWT_EXPIRES_IN: z.string().default('24h'),
  SESSION_SECRET: z.string()
    .min(32, 'SESSION_SECRET must be at least 32 characters long')
    .optional()
    .default('your-super-secret-session-key-that-is-at-least-32-characters-long'),
  
  // Database
  DATABASE_URL: z.string()
    .url('DATABASE_URL must be a valid connection string')
    .refine(url => {
      if (process.env.NODE_ENV === 'production') {
        return !url.includes('localhost') && !url.includes('127.0.0.1');
      }
      return true;
    }, 'DATABASE_URL must point to a production database in production'),
  
  // Supabase
  SUPABASE_URL: z.string()
    .url('SUPABASE_URL must be a valid URL'),
  SUPABASE_ANON_KEY: z.string()
    .min(32, 'SUPABASE_ANON_KEY must be at least 32 characters long'),
  SUPABASE_SERVICE_ROLE_KEY: z.string()
    .min(32, 'SUPABASE_SERVICE_ROLE_KEY must be at least 32 characters long'),
  
  // WebSocket
  VITE_WS_URL: z.string()
    .url('VITE_WS_URL must be a valid WebSocket URL')
    .refine(url => url.startsWith('ws://') || url.startsWith('wss://'), {
      message: 'VITE_WS_URL must start with ws:// or wss://'
    }),
  
  // CORS
  CORS_ORIGIN: z.string()
    .transform(origins => origins.split(',').map(o => o.trim()))
    .refine(origins => {
      if (process.env.NODE_ENV === 'production') {
        return !origins.some(origin => 
          origin.includes('localhost') || 
          origin.includes('127.0.0.1')
        );
      }
      return true;
    }, 'CORS_ORIGIN should not include localhost in production'),
  
  // Python AI Engine
  PYTHON_SERVER_URL: z.string()
    .url('PYTHON_SERVER_URL must be a valid URL')
    .default('http://localhost:5001'),
  PYTHON_SERVER_PORT: z.string()
    .regex(/^\d+$/, 'PYTHON_SERVER_PORT must be a number')
    .default('5001'),
  PYTHON_HEALTH_CHECK_URL: z.string()
    .url('PYTHON_HEALTH_CHECK_URL must be a valid URL')
    .default('http://localhost:5001/health'),
  
  // Optional Features
  ENABLE_METRICS: z.string()
    .default('true')
    .transform((val) => val === 'true')
    .pipe(z.boolean()),
  ENABLE_DEV_TOOLS: z.string()
    .default('false')
    .transform((val) => val === 'true')
    .pipe(z.boolean())
    .refine((val) => process.env.NODE_ENV !== 'production' || val === false, {
      message: 'Dev tools cannot be enabled in production'
    })
});

// Validate environment variables
export function validateEnvironment(): boolean {
  try {
    const result = envSchema.safeParse(process.env);
    
    if (!result.success) {
      const errorDetails = result.error.issues.map(issue => ({
        path: issue.path.join('.'),
        message: issue.message
      }));
      
      logger.error('❌ Environment validation failed:');
      errorDetails.forEach(({ path, message }) => {
        logger.error(`- ${path}: ${message}`);
      });
      return false;
    }
    
    logger.info('✅ Environment variables validated successfully');
    return true;
  } catch (error) {
    logger.error('❌ Error validating environment variables:', error);
    return false;
  }
}

// Type-safe configuration object
export const config = (() => {
  const env = envSchema.parse(process.env);
  
  return {
    // Core
    env: env.NODE_ENV,
    port: parseInt(env.PORT, 10),
    isProduction: env.NODE_ENV === 'production',
    isDevelopment: env.NODE_ENV === 'development',
    isTest: env.NODE_ENV === 'test',
    
    // API
    api: {
      baseUrl: env.API_BASE_URL,
      version: 'v2',
      fullUrl: `${env.API_BASE_URL}/api/v2`
    },
    
    // Security
    jwt: {
      secret: env.JWT_SECRET,
      algorithm: env.JWT_ALGORITHM as 'HS256',
      expiresIn: env.JWT_EXPIRES_IN
    },
    session: {
      secret: env.SESSION_SECRET
    },
    
    // Database
    database: {
      url: env.DATABASE_URL
    },
    
    // Supabase
    supabase: {
      url: env.SUPABASE_URL,
      anonKey: env.SUPABASE_ANON_KEY,
      serviceRoleKey: env.SUPABASE_SERVICE_ROLE_KEY
    },
    
    // WebSocket
    ws: {
      url: env.VITE_WS_URL
    },
    
    // CORS
    cors: {
      origin: Array.isArray(env.CORS_ORIGIN) ? env.CORS_ORIGIN : [env.CORS_ORIGIN],
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      credentials: true
    },
    
    // Python AI Engine
    python: {
      serverUrl: env.PYTHON_SERVER_URL,
      serverPort: parseInt(env.PYTHON_SERVER_PORT, 10),
      healthCheckUrl: env.PYTHON_HEALTH_CHECK_URL
    },
    
    // Features
    features: {
      metrics: env.ENABLE_METRICS,
      devTools: env.ENABLE_DEV_TOOLS
    }
  } as const;
})();

export default config;
