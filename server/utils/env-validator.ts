import { config } from 'dotenv';
import { z } from 'zod';
import { logger } from '../config/logger';

// Define the schema for environment variables
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.string().default('5001'),
  DATABASE_URL: z.string().min(1, 'DATABASE_URL is required'),
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters'),
  JWT_EXPIRES_IN: z.string().default('1d'),
  PYTHON_SERVICE_URL: z.string().url().default('http://localhost:8000'),
  CORS_ORIGIN: z.string().default('*'),
  // Add more environment variables as needed
});

type EnvVars = z.infer<typeof envSchema>;

// Load environment variables
const envVars = config().parsed;

// Validate environment variables
export const validateEnvironment = (): EnvVars => {
  try {
    const validatedVars = envSchema.parse(process.env);
    
    // Set process.env with validated values
    Object.entries(validatedVars).forEach(([key, value]) => {
      if (process.env[key] === undefined) {
        process.env[key] = value;
      }
    });
    
    return validatedVars;
  } catch (error: unknown) {
    if (error instanceof z.ZodError) {
      const errorMessage = 'Invalid environment variables:';
      const errorDetails = error.issues.map(issue => ({
        path: issue.path.join('.'),
        message: issue.message,
        code: issue.code
      }));
      
      logger.error(errorMessage, { errors: errorDetails });
      throw new Error(`${errorMessage}\n${JSON.stringify(errorDetails, null, 2)}`);
    }
    
    logger.error('Failed to validate environment variables', { error });
    throw new Error('Failed to validate environment variables');
  }
};

// Export validated environment variables
export const env = envSchema.parse(process.env);

export default env;
