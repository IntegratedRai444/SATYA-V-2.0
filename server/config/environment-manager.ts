import { logger } from './logger';
import fs from 'fs';
import path from 'path';

export interface EnvironmentConfig {
  name: string;
  database: {
    url: string;
    type: 'sqlite' | 'postgresql' | 'mysql';
    ssl?: boolean;
    poolSize?: number;
    connectionTimeout?: number;
  };
  server: {
    port: number;
    host: string;
    cors: {
      origins: string[];
      credentials: boolean;
    };
    rateLimit: {
      windowMs: number;
      maxRequests: number;
    };
  };
  python: {
    serverUrl: string;
    timeout: number;
    retries: number;
  };
  storage: {
    uploadDir: string;
    maxFileSize: string;
    allowedTypes: string[];
  };
  logging: {
    level: string;
    dir: string;
    maxFiles: number;
    maxSize: string;
  };
  security: {
    jwtSecret: string;
    sessionSecret: string;
    bcryptRounds: number;
    csrfEnabled: boolean;
    httpsOnly: boolean;
  };
  monitoring: {
    metricsEnabled: boolean;
    alertingEnabled: boolean;
    healthCheckInterval: number;
  };
  features: {
    registration: boolean;
    guestAccess: boolean;
    fileUpload: boolean;
    realTimeUpdates: boolean;
  };
  external: {
    redis?: {
      url: string;
      password?: string;
    };
    email?: {
      smtp: {
        host: string;
        port: number;
        secure: boolean;
        auth: {
          user: string;
          pass: string;
        };
      };
    };
    webhook?: {
      url: string;
      secret: string;
    };
  };
}

class EnvironmentManager {
  private config: EnvironmentConfig;
  private environment: string;

  constructor() {
    this.environment = process.env.NODE_ENV || 'development';
    this.config = this.loadConfiguration();
    this.validateConfiguration();
  }

  private loadConfiguration(): EnvironmentConfig {
    // Load base configuration
    const baseConfig = this.getBaseConfiguration();
    
    // Load environment-specific overrides
    const envConfig = this.loadEnvironmentOverrides();
    
    // Merge configurations
    return this.mergeConfigurations(baseConfig, envConfig);
  }

  private getBaseConfiguration(): EnvironmentConfig {
    return {
      name: this.environment,
      database: {
        url: process.env.DATABASE_URL || 'sqlite:./data/satyaai.db',
        type: this.getDatabaseType(),
        ssl: process.env.DATABASE_SSL === 'true',
        poolSize: parseInt(process.env.DATABASE_POOL_SIZE || '10'),
        connectionTimeout: parseInt(process.env.DATABASE_TIMEOUT || '30000')
      },
      server: {
        port: parseInt(process.env.PORT || '3000'),
        host: process.env.HOST || '0.0.0.0',
        cors: {
          origins: this.parseCorsOrigins(),
          credentials: process.env.CORS_CREDENTIALS !== 'false'
        },
        rateLimit: {
          windowMs: parseInt(process.env.RATE_LIMIT_WINDOW || '900000'), // 15 minutes
          maxRequests: parseInt(process.env.RATE_LIMIT_MAX || '100')
        }
      },
      python: {
        serverUrl: process.env.PYTHON_SERVER_URL || 'http://localhost:5001',
        timeout: parseInt(process.env.PYTHON_TIMEOUT || '30000'),
        retries: parseInt(process.env.PYTHON_RETRIES || '3')
      },
      storage: {
        uploadDir: process.env.UPLOAD_DIR || './uploads',
        maxFileSize: process.env.MAX_FILE_SIZE || '100MB',
        allowedTypes: this.parseAllowedTypes()
      },
      logging: {
        level: process.env.LOG_LEVEL || 'info',
        dir: process.env.LOG_DIR || './logs',
        maxFiles: parseInt(process.env.LOG_MAX_FILES || '10'),
        maxSize: process.env.LOG_MAX_SIZE || '10MB'
      },
      security: {
        jwtSecret: process.env.JWT_SECRET || this.generateSecret(),
        sessionSecret: process.env.SESSION_SECRET || this.generateSecret(),
        bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS || '12'),
        csrfEnabled: process.env.CSRF_ENABLED !== 'false',
        httpsOnly: process.env.HTTPS_ONLY === 'true'
      },
      monitoring: {
        metricsEnabled: process.env.METRICS_ENABLED !== 'false',
        alertingEnabled: process.env.ALERTING_ENABLED !== 'false',
        healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL || '30000')
      },
      features: {
        registration: process.env.FEATURE_REGISTRATION !== 'false',
        guestAccess: process.env.FEATURE_GUEST_ACCESS === 'true',
        fileUpload: process.env.FEATURE_FILE_UPLOAD !== 'false',
        realTimeUpdates: process.env.FEATURE_REALTIME !== 'false'
      },
      external: {
        redis: process.env.REDIS_URL ? {
          url: process.env.REDIS_URL,
          password: process.env.REDIS_PASSWORD
        } : undefined,
        email: process.env.SMTP_HOST ? {
          smtp: {
            host: process.env.SMTP_HOST,
            port: parseInt(process.env.SMTP_PORT || '587'),
            secure: process.env.SMTP_SECURE === 'true',
            auth: {
              user: process.env.SMTP_USER || '',
              pass: process.env.SMTP_PASS || ''
            }
          }
        } : undefined,
        webhook: process.env.WEBHOOK_URL ? {
          url: process.env.WEBHOOK_URL,
          secret: process.env.WEBHOOK_SECRET || ''
        } : undefined
      }
    };
  }

  private loadEnvironmentOverrides(): Partial<EnvironmentConfig> {
    const configPath = path.join(process.cwd(), 'config', `${this.environment}.json`);
    
    try {
      if (fs.existsSync(configPath)) {
        const configFile = fs.readFileSync(configPath, 'utf8');
        return JSON.parse(configFile);
      }
    } catch (error) {
      logger.warn(`Failed to load environment config from ${configPath}:`, error);
    }
    
    return {};
  }

  private mergeConfigurations(base: EnvironmentConfig, override: Partial<EnvironmentConfig>): EnvironmentConfig {
    return {
      ...base,
      ...override,
      database: { ...base.database, ...override.database },
      server: { 
        ...base.server, 
        ...override.server,
        cors: { ...base.server.cors, ...override.server?.cors },
        rateLimit: { ...base.server.rateLimit, ...override.server?.rateLimit }
      },
      python: { ...base.python, ...override.python },
      storage: { ...base.storage, ...override.storage },
      logging: { ...base.logging, ...override.logging },
      security: { ...base.security, ...override.security },
      monitoring: { ...base.monitoring, ...override.monitoring },
      features: { ...base.features, ...override.features },
      external: {
        ...base.external,
        ...override.external,
        redis: { ...base.external.redis, ...override.external?.redis },
        email: override.external?.email ? {
          smtp: { ...base.external.email?.smtp, ...override.external.email.smtp }
        } : base.external.email,
        webhook: { ...base.external.webhook, ...override.external?.webhook }
      }
    };
  }

  private validateConfiguration(): void {
    const errors: string[] = [];

    // Validate required fields
    if (!this.config.security.jwtSecret || this.config.security.jwtSecret.length < 32) {
      errors.push('JWT_SECRET must be at least 32 characters long');
    }

    if (!this.config.security.sessionSecret || this.config.security.sessionSecret.length < 32) {
      errors.push('SESSION_SECRET must be at least 32 characters long');
    }

    if (this.config.server.port < 1 || this.config.server.port > 65535) {
      errors.push('PORT must be between 1 and 65535');
    }

    if (!this.config.python.serverUrl.startsWith('http')) {
      errors.push('PYTHON_SERVER_URL must be a valid HTTP URL');
    }

    // Production-specific validations
    if (this.environment === 'production') {
      if (this.config.security.jwtSecret === 'default-secret') {
        errors.push('JWT_SECRET must be changed from default in production');
      }

      if (this.config.security.sessionSecret === 'default-secret') {
        errors.push('SESSION_SECRET must be changed from default in production');
      }

      if (!this.config.security.httpsOnly) {
        logger.warn('HTTPS_ONLY is not enabled in production environment');
      }

      if (this.config.database.type === 'sqlite') {
        logger.warn('SQLite is not recommended for production use');
      }
    }

    if (errors.length > 0) {
      throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
    }
  }

  private getDatabaseType(): 'sqlite' | 'postgresql' | 'mysql' {
    const url = process.env.DATABASE_URL || '';
    
    if (url.startsWith('postgresql://') || url.startsWith('postgres://')) {
      return 'postgresql';
    }
    
    if (url.startsWith('mysql://')) {
      return 'mysql';
    }
    
    return 'sqlite';
  }

  private parseCorsOrigins(): string[] {
    const origins = process.env.CORS_ORIGINS || 'http://localhost:3000,http://localhost:5173';
    return origins.split(',').map(origin => origin.trim());
  }

  private parseAllowedTypes(): string[] {
    const types = process.env.ALLOWED_FILE_TYPES || 'jpg,jpeg,png,gif,mp4,avi,mov,mp3,wav,flac';
    return types.split(',').map(type => type.trim());
  }

  private generateSecret(): string {
    if (this.environment === 'production') {
      throw new Error('Secret keys must be explicitly set in production');
    }
    
    return 'default-secret-' + Math.random().toString(36).substring(2, 15);
  }

  // Public methods
  getConfig(): EnvironmentConfig {
    return { ...this.config };
  }

  getEnvironment(): string {
    return this.environment;
  }

  isProduction(): boolean {
    return this.environment === 'production';
  }

  isDevelopment(): boolean {
    return this.environment === 'development';
  }

  isTest(): boolean {
    return this.environment === 'test';
  }

  updateConfig(updates: Partial<EnvironmentConfig>): void {
    this.config = this.mergeConfigurations(this.config, updates);
    this.validateConfiguration();
    logger.info('Configuration updated', { environment: this.environment });
  }

  // Get specific configuration sections
  getDatabaseConfig() {
    return this.config.database;
  }

  getServerConfig() {
    return this.config.server;
  }

  getPythonConfig() {
    return this.config.python;
  }

  getStorageConfig() {
    return this.config.storage;
  }

  getLoggingConfig() {
    return this.config.logging;
  }

  getSecurityConfig() {
    return this.config.security;
  }

  getMonitoringConfig() {
    return this.config.monitoring;
  }

  getFeaturesConfig() {
    return this.config.features;
  }

  getExternalConfig() {
    return this.config.external;
  }

  // Configuration summary for logging
  getConfigSummary(): Record<string, any> {
    return {
      environment: this.environment,
      database: {
        type: this.config.database.type,
        ssl: this.config.database.ssl
      },
      server: {
        port: this.config.server.port,
        host: this.config.server.host
      },
      features: this.config.features,
      monitoring: {
        metricsEnabled: this.config.monitoring.metricsEnabled,
        alertingEnabled: this.config.monitoring.alertingEnabled
      },
      external: {
        redis: !!this.config.external.redis,
        email: !!this.config.external.email,
        webhook: !!this.config.external.webhook
      }
    };
  }
}

// Singleton instance
const environmentManager = new EnvironmentManager();

export default environmentManager;
export { EnvironmentManager };