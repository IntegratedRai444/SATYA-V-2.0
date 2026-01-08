import { config, ConfigurationError } from './environment';
import { logger } from './logger';

/**
 * Validates that all required environment variables are set
 */
function validateEnvironment() {
  const requiredVars = [
    'NODE_ENV',
    'PORT',
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY',
    'SUPABASE_SERVICE_ROLE_KEY',
    'JWT_SECRET',
    'DATABASE_URL'
  ];

  const missingVars = requiredVars.filter(varName => !(varName in config));

  if (missingVars.length > 0) {
    throw new ConfigurationError(
      `Missing required environment variables: ${missingVars.join(', ')}`
    );
  }

  // Validate API version consistency
  if (process.env.NODE_ENV === 'production') {
    if (process.env.VITE_API_URL && !process.env.VITE_API_URL.includes('/api/v2')) {
      logger.warn('VITE_API_URL in production should include /api/v2 prefix');
    }
  }

  logger.info('Environment configuration validated successfully');
}

export { validateEnvironment };
