import { logger } from '../config';

interface EnvConfig {
  name: string;
  required: boolean;
  defaultValue?: string;
  description?: string;
}

const requiredEnvVars: EnvConfig[] = [
  {
    name: 'SUPABASE_URL',
    required: true,
    description: 'Supabase project URL'
  },
  {
    name: 'SUPABASE_ANON_KEY',
    required: true,
    description: 'Supabase anonymous key'
  },
  {
    name: 'DATABASE_URL',
    required: true,
    description: 'PostgreSQL connection string'
  },
  {
    name: 'JWT_SECRET',
    required: true,
    description: 'JWT signing secret'
  },
  {
    name: 'SESSION_SECRET',
    required: true,
    description: 'Session encryption secret'
  },
  {
    name: 'NODE_ENV',
    required: false,
    defaultValue: 'development',
    description: 'Application environment'
  },
  {
    name: 'PORT',
    required: false,
    defaultValue: '3000',
    description: 'Server port'
  },
  {
    name: 'PYTHON_API_URL',
    required: false,
    defaultValue: 'http://localhost:5001',
    description: 'Python AI server URL'
  }
];

/**
 * Validate all required environment variables are set
 * @returns true if all required vars are present, false otherwise
 */
export function validateEnvironment(): boolean {
  const missing: string[] = [];
  const warnings: string[] = [];

  for (const config of requiredEnvVars) {
    const value = process.env[config.name];

    if (!value) {
      if (config.required) {
        missing.push(`${config.name}${config.description ? ` (${config.description})` : ''}`);
      } else if (config.defaultValue) {
        warnings.push(
          `${config.name} not set, using default: ${config.defaultValue}`
        );
        process.env[config.name] = config.defaultValue;
      }
    }
  }

  // Log warnings
  if (warnings.length > 0) {
    logger.warn('Environment variable warnings:', { warnings });
  }

  // Log errors
  if (missing.length > 0) {
    logger.error('Missing required environment variables:', { missing });
    logger.error('Please set these variables in your .env file or environment');
    return false;
  }

  logger.info('✅ Environment validation passed');
  return true;
}

/**
 * Get environment variable with type safety
 */
export function getEnv(name: string, defaultValue?: string): string {
  const value = process.env[name];
  if (!value && !defaultValue) {
    throw new Error(`Environment variable ${name} is not set and no default provided`);
  }
  return value || defaultValue!;
}

/**
 * Get environment variable as number
 */
export function getEnvNumber(name: string, defaultValue?: number): number {
  const value = process.env[name];
  if (!value) {
    if (defaultValue === undefined) {
      throw new Error(`Environment variable ${name} is not set and no default provided`);
    }
    return defaultValue;
  }
  const parsed = parseInt(value, 10);
  if (isNaN(parsed)) {
    throw new Error(`Environment variable ${name} is not a valid number: ${value}`);
  }
  return parsed;
}

/**
 * Get environment variable as boolean
 */
export function getEnvBoolean(name: string, defaultValue?: boolean): boolean {
  const value = process.env[name];
  if (!value) {
    if (defaultValue === undefined) {
      throw new Error(`Environment variable ${name} is not set and no default provided`);
    }
    return defaultValue;
  }
  return value.toLowerCase() === 'true' || value === '1';
}
