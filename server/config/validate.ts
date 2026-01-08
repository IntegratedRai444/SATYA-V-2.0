import * as Joi from 'joi';
import { logger } from './logger';

const configSchema = Joi.object({
  NODE_ENV: Joi.string()
    .valid('development', 'production', 'test')
    .default('development'),
  PORT: Joi.number().default(3000),
  JWT_SECRET: Joi.string().required(),
  DATABASE_URL: Joi.string().required(),
  REDIS_URL: Joi.string().uri().required(),
  RATE_LIMIT_WINDOW_MS: Joi.number().default(60000),
  RATE_LIMIT_MAX: Joi.number().default(100),
});

export function validateConfig(config: NodeJS.ProcessEnv): void {
  const { error, value } = configSchema.validate(config, {
    abortEarly: false,
    allowUnknown: true,
  });

  if (error) {
    logger.error('Invalid configuration:', error.details);
    process.exit(1);
  }

  // Set validated config back to process.env
  Object.assign(process.env, value);
}
