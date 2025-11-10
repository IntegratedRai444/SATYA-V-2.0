import { z } from 'zod';
import rateLimit, { ipKeyGenerator } from 'express-rate-limit';
import helmet from 'helmet';

// Environment validation schema
export const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  JWT_SECRET: z.string().min(32, 'JWT secret must be at least 32 characters'),
  JWT_EXPIRES_IN: z.string().default('24h'),
  CORS_ORIGIN: z.string().default('http://localhost:3000'),
  RATE_LIMIT_WINDOW_MS: z.string().transform(Number).default('900000'), // 15 minutes
  RATE_LIMIT_MAX_REQUESTS: z.string().transform(Number).default('100'),
  RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS: z.string().transform(Boolean).default('false'),
  MAX_FILE_SIZE: z.string().transform(Number).default('52428800'), // 50MB
  SESSION_SECRET: z.string().min(32, 'Session secret must be at least 32 characters'),
});

// Validate environment variables
export const env = envSchema.parse(process.env);

// CORS configuration
export const corsConfig = {
  origin: env.CORS_ORIGIN.split(',').map(origin => origin.trim()),
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: [
    'Origin',
    'X-Requested-With',
    'Content-Type',
    'Accept',
    'Authorization',
    'X-API-Key'
  ],
  credentials: true,
  maxAge: 86400, // 24 hours
  optionsSuccessStatus: 200
};

// Rate limiting configurations
export const rateLimitConfig = {
  // General API rate limiting
  general: rateLimit({
    windowMs: env.RATE_LIMIT_WINDOW_MS,
    max: env.RATE_LIMIT_MAX_REQUESTS,
    message: {
      error: 'Too many requests from this IP, please try again later.',
      retryAfter: Math.ceil(env.RATE_LIMIT_WINDOW_MS / 1000)
    },
    standardHeaders: true,
    legacyHeaders: false,
    skipSuccessfulRequests: env.RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS,
    // Use default keyGenerator to avoid IPv6 issues
  }),

  // Stricter rate limiting for authentication endpoints
  auth: rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts per 15 minutes
    message: {
      error: 'Too many authentication attempts, please try again later.',
      retryAfter: 900
    },
    standardHeaders: true,
    legacyHeaders: false,
    skipSuccessfulRequests: true
  }),

  // Rate limiting for file uploads
  upload: rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 10, // 10 uploads per hour
    message: {
      error: 'Too many file uploads, please try again later.',
      retryAfter: 3600
    },
    standardHeaders: true,
    legacyHeaders: false,
    skipSuccessfulRequests: false
  }),

  // Rate limiting for analysis endpoints
  analysis: rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 20, // 20 analysis requests per hour
    message: {
      error: 'Too many analysis requests, please try again later.',
      retryAfter: 3600
    },
    standardHeaders: true,
    legacyHeaders: false,
    skipSuccessfulRequests: false
  })
};

// Helmet security headers configuration
export const helmetConfig = {
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "https:"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      connectSrc: ["'self'", "ws:", "wss:"],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
      upgradeInsecureRequests: env.NODE_ENV === 'production' ? [] : null
    }
  },
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: "cross-origin" as const }
};

// Input validation schemas
export const validationSchemas = {
  // Authentication schemas
  login: z.object({
    username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_-]+$/, 'Username contains invalid characters'),
    password: z.string().min(8).max(128)
  }),

  register: z.object({
    username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_-]+$/, 'Username contains invalid characters'),
    email: z.string().email(),
    password: z.string().min(8).max(128).regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
    )
  }),

  // Analysis schemas
  imageAnalysis: z.object({
    imageData: z.string().optional(),
    analysisType: z.enum(['comprehensive', 'quick', 'detailed']).default('comprehensive'),
    confidenceThreshold: z.number().min(0).max(100).default(80),
    enableAdvancedModels: z.boolean().default(true)
  }),

  videoAnalysis: z.object({
    analysisType: z.enum(['comprehensive', 'quick', 'detailed']).default('comprehensive'),
    confidenceThreshold: z.number().min(0).max(100).default(80),
    enableAdvancedModels: z.boolean().default(true),
    frameRate: z.number().min(1).max(60).default(30)
  }),

  audioAnalysis: z.object({
    analysisType: z.enum(['comprehensive', 'quick', 'detailed']).default('comprehensive'),
    confidenceThreshold: z.number().min(0).max(100).default(80),
    enableAdvancedModels: z.boolean().default(true)
  }),

  // Settings schemas
  userPreferences: z.object({
    theme: z.enum(['light', 'dark', 'auto']).default('auto'),
    language: z.string().length(2).default('en'),
    confidenceThreshold: z.number().min(0).max(100).default(80),
    enableNotifications: z.boolean().default(true),
    autoAnalyze: z.boolean().default(false),
    sensitivityLevel: z.enum(['low', 'medium', 'high']).default('medium')
  }),

  userProfile: z.object({
    email: z.string().email().optional(),
    name: z.string().min(2).max(100).optional(),
    bio: z.string().max(500).optional()
  }),

  // File upload schemas
  fileUpload: z.object({
    maxFileSize: z.number().max(env.MAX_FILE_SIZE),
    allowedTypes: z.array(z.string()),
    maxFiles: z.number().max(10)
  })
};

// JWT configuration
export const jwtConfig = {
  secret: env.JWT_SECRET,
  expiresIn: env.JWT_EXPIRES_IN,
  algorithm: 'HS256' as const,
  issuer: 'satyaai-backend',
  audience: 'satyaai-frontend'
};

// File upload configuration
export const uploadConfig = {
  maxFileSize: env.MAX_FILE_SIZE,
  allowedImageTypes: ['image/jpeg', 'image/png', 'image/webp', 'image/gif'],
  allowedVideoTypes: ['video/mp4', 'video/avi', 'video/mov', 'video/webm'],
  allowedAudioTypes: ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp3'],
  maxFiles: 10
};

// Password policy
export const passwordPolicy = {
  minLength: 8,
  maxLength: 128,
  requireUppercase: true,
  requireLowercase: true,
  requireNumbers: true,
  requireSpecialChars: true,
  preventCommonPasswords: true
};

// Session configuration
export const sessionConfig = {
  secret: env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    sameSite: 'strict' as const
  }
}; 