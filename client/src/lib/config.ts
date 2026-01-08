/**
 * Centralized Configuration Management System
 * Provides type-safe, validated configuration for the entire application
 */

import { z } from 'zod';
import logger from './logger';

// ============================================================================
// Configuration Schemas
// ============================================================================

const envSchema = z.object({
  // API Configuration
  VITE_API_URL: z.string().url().default('http://localhost:5001'),
  VITE_API_TIMEOUT: z.coerce.number().default(300000), // 5 minutes
  
  // WebSocket Configuration
  VITE_WS_URL: z.string().url().optional(),
  VITE_WS_RECONNECT_ATTEMPTS: z.coerce.number().default(5),
  VITE_WS_RECONNECT_DELAY: z.coerce.number().default(3000),
  
  // Feature Flags
  VITE_ENABLE_ANALYTICS: z.coerce.boolean().default(true),
  VITE_ENABLE_NOTIFICATIONS: z.coerce.boolean().default(true),
  VITE_ENABLE_WEBCAM: z.coerce.boolean().default(true),
  VITE_ENABLE_BATCH_UPLOAD: z.coerce.boolean().default(true),
  
  // Logging Configuration
  VITE_LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  VITE_ENABLE_REMOTE_LOGGING: z.coerce.boolean().default(false),
  VITE_LOG_ENDPOINT: z.string().url().optional(),
  
  // Upload Configuration
  VITE_MAX_FILE_SIZE: z.coerce.number().default(10485760), // 10MB
  VITE_MAX_BATCH_SIZE: z.coerce.number().default(10),
  VITE_ALLOWED_IMAGE_TYPES: z.string().default('image/jpeg,image/png,image/webp'),
  VITE_ALLOWED_VIDEO_TYPES: z.string().default('video/mp4,video/webm'),
  VITE_ALLOWED_AUDIO_TYPES: z.string().default('audio/mp3,audio/wav,audio/ogg'),
  
  // Performance Configuration
  VITE_ENABLE_LAZY_LOADING: z.coerce.boolean().default(true),
  VITE_ENABLE_CODE_SPLITTING: z.coerce.boolean().default(true),
  VITE_IMAGE_OPTIMIZATION: z.coerce.boolean().default(true),
  
  // Environment
  MODE: z.enum(['development', 'production', 'test']).default('development'),
  DEV: z.coerce.boolean().default(true),
  PROD: z.coerce.boolean().default(false),
});

type EnvConfig = z.infer<typeof envSchema>;

// ============================================================================
// Application Configuration Interface
// ============================================================================

export interface AppConfig {
  api: {
    baseUrl: string;
    timeout: number;
    retryAttempts: number;
  };
  websocket: {
    url: string;
    reconnect: boolean;
    maxReconnectAttempts: number;
    reconnectDelay: number;
  };
  features: {
    enableAnalytics: boolean;
    enableNotifications: boolean;
    enableWebcam: boolean;
    enableBatchUpload: boolean;
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    enableRemote: boolean;
    remoteEndpoint?: string;
  };
  upload: {
    maxFileSize: number;
    maxBatchSize: number;
    allowedTypes: {
      image: string[];
      video: string[];
      audio: string[];
    };
  };
  performance: {
    enableLazyLoading: boolean;
    enableCodeSplitting: boolean;
    imageOptimization: boolean;
  };
  environment: 'development' | 'production' | 'test';
  isDevelopment: boolean;
  isProduction: boolean;
}

// ============================================================================
// Configuration Manager Class
// ============================================================================

class ConfigManager {
  private config: AppConfig | null = null;
  private envConfig: EnvConfig | null = null;
  private validationErrors: string[] = [];

  /**
   * Load and validate configuration
   */
  async load(): Promise<void> {
    try {
      // Parse and validate environment variables
      const result = envSchema.safeParse(import.meta.env);
      
      if (!result.success) {
        this.validationErrors = result.error.errors.map(
          err => `${err.path.join('.')}: ${err.message}`
        );
        logger.warn('Configuration validation failed', { errors: this.validationErrors });
        
        // Use defaults for failed validation
        this.envConfig = envSchema.parse({});
      } else {
        this.envConfig = result.data;
        logger.debug('Configuration loaded successfully');
      }

      // Build application configuration
      this.config = this.buildAppConfig(this.envConfig);
      
      // Log configuration in development
      if (this.config.isDevelopment) {
        logger.debug('Application configuration', { config: this.config });
      }
    } catch (error) {
      logger.error('Failed to load configuration', error as Error);
      throw new Error('Configuration initialization failed');
    }
  }

  /**
   * Build application configuration from environment config
   */
  private buildAppConfig(envConfig: EnvConfig): AppConfig {
    const wsUrl = envConfig.VITE_WS_URL || 
      envConfig.VITE_API_URL.replace(/^http/, 'ws').replace(/:\d+/, ':5001');

    return {
      api: {
        baseUrl: envConfig.VITE_API_URL,
        timeout: envConfig.VITE_API_TIMEOUT,
        retryAttempts: 3,
      },
      websocket: {
        url: wsUrl,
        reconnect: true,
        maxReconnectAttempts: envConfig.VITE_WS_RECONNECT_ATTEMPTS,
        reconnectDelay: envConfig.VITE_WS_RECONNECT_DELAY,
      },
      features: {
        enableAnalytics: envConfig.VITE_ENABLE_ANALYTICS,
        enableNotifications: envConfig.VITE_ENABLE_NOTIFICATIONS,
        enableWebcam: envConfig.VITE_ENABLE_WEBCAM,
        enableBatchUpload: envConfig.VITE_ENABLE_BATCH_UPLOAD,
      },
      logging: {
        level: envConfig.VITE_LOG_LEVEL,
        enableRemote: envConfig.VITE_ENABLE_REMOTE_LOGGING,
        remoteEndpoint: envConfig.VITE_LOG_ENDPOINT,
      },
      upload: {
        maxFileSize: envConfig.VITE_MAX_FILE_SIZE,
        maxBatchSize: envConfig.VITE_MAX_BATCH_SIZE,
        allowedTypes: {
          image: envConfig.VITE_ALLOWED_IMAGE_TYPES.split(','),
          video: envConfig.VITE_ALLOWED_VIDEO_TYPES.split(','),
          audio: envConfig.VITE_ALLOWED_AUDIO_TYPES.split(','),
        },
      },
      performance: {
        enableLazyLoading: envConfig.VITE_ENABLE_LAZY_LOADING,
        enableCodeSplitting: envConfig.VITE_ENABLE_CODE_SPLITTING,
        imageOptimization: envConfig.VITE_IMAGE_OPTIMIZATION,
      },
      environment: envConfig.MODE,
      isDevelopment: envConfig.DEV,
      isProduction: envConfig.PROD,
    };
  }

  /**
   * Get configuration value by key path
   */
  get<K extends keyof AppConfig>(key: K): AppConfig[K] {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call load() first.');
    }
    return this.config[key];
  }

  /**
   * Get entire configuration
   */
  getAll(): Readonly<AppConfig> {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call load() first.');
    }
    return { ...this.config };
  }

  /**
   * Validate configuration
   */
  validate(): { valid: boolean; errors: string[] } {
    const errors: string[] = [...this.validationErrors];

    if (!this.config) {
      errors.push('Configuration not loaded');
      return { valid: false, errors };
    }

    // Additional runtime validations
    if (this.config.api.timeout < 1000) {
      errors.push('API timeout must be at least 1000ms');
    }

    if (this.config.upload.maxFileSize < 1024) {
      errors.push('Max file size must be at least 1KB');
    }

    if (this.config.websocket.maxReconnectAttempts < 1) {
      errors.push('Max reconnect attempts must be at least 1');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Check if configuration is loaded
   */
  isLoaded(): boolean {
    return this.config !== null;
  }

  /**
   * Get validation errors
   */
  getValidationErrors(): string[] {
    return [...this.validationErrors];
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

const configManager = new ConfigManager();

// ============================================================================
// Exported Functions
// ============================================================================

/**
 * Initialize configuration (must be called before using config)
 */
export async function initializeConfig(): Promise<void> {
  await configManager.load();
  
  const validation = configManager.validate();
  if (!validation.valid) {
    logger.warn('Configuration validation warnings', { errors: validation.errors });
  }
}

/**
 * Get configuration value
 */
export function getConfig<K extends keyof AppConfig>(key: K): AppConfig[K] {
  return configManager.get(key);
}

/**
 * Get all configuration
 */
export function getAllConfig(): Readonly<AppConfig> {
  return configManager.getAll();
}

/**
 * Check if configuration is valid
 */
export function validateConfig(): { valid: boolean; errors: string[] } {
  return configManager.validate();
}

/**
 * Check if configuration is loaded
 */
export function isConfigLoaded(): boolean {
  return configManager.isLoaded();
}

// ============================================================================
// Legacy Compatibility (Deprecated)
// ============================================================================

/** @deprecated Use getConfig('api').baseUrl instead */
export async function getServerUrl(): Promise<string> {
  if (!configManager.isLoaded()) {
    await configManager.load();
  }
  return configManager.get('api').baseUrl;
}

/** @deprecated Use getConfig('api').baseUrl instead */
export async function getServerConfig(): Promise<{ server_url: string; port: number; timestamp: string }> {
  if (!configManager.isLoaded()) {
    await configManager.load();
  }
  const apiConfig = configManager.get('api');
  return {
    server_url: apiConfig.baseUrl,
    port: 3000,
    timestamp: new Date().toISOString(),
  };
}

/** @deprecated No longer needed */
export function clearServerConfigCache(): void {
  logger.warn('clearServerConfigCache is deprecated and does nothing');
}

/** @deprecated Use getConfig('api').baseUrl + endpoint instead */
export async function createApiUrl(endpoint: string): Promise<string> {
  if (!configManager.isLoaded()) {
    await configManager.load();
  }
  return `${configManager.get('api').baseUrl}${endpoint}`;
}

// Export the manager for testing
export { configManager, ConfigManager };

// Export default
export default configManager; 