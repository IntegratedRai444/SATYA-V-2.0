import path from 'path';
import { z } from 'zod';

// Define model configuration schema
const modelConfigSchema = z.object({
  // Base paths
  modelsDir: z.string().default(path.join(process.cwd(), 'models')),
  cacheDir: z.string().default(path.join(process.cwd(), '.cache')),
  
  // Model specific configurations
  imageModel: z.object({
    name: z.string().default('efficientnet-b7'),
    version: z.string().default('1.0.0'),
    inputSize: z.tuple([z.number(), z.number()]).default([224, 224]),
    batchSize: z.number().default(32),
  }),
  
  videoModel: z.object({
    name: z.string().default('x3d_xs'),
    version: z.string().default('1.0.0'),
    frameCount: z.number().default(16),
    fps: z.number().default(30),
  }),
  
  audioModel: z.object({
    name: z.string().default('wav2vec2-base'),
    version: z.string().default('1.0.0'),
    sampleRate: z.number().default(16000),
    duration: z.number().default(5),
  }),
  
  // Performance settings
  performance: z.object({
    useGpu: z.boolean().default(true),
    maxWorkers: z.number().default(4),
    maxBatchSize: z.number().default(8),
    cacheModels: z.boolean().default(true),
  }),
  
  // Model download settings
  download: z.object({
    baseUrl: z.string().default('https://models.satyaai.tech'),
    timeout: z.number().default(300000), // 5 minutes
    retries: z.number().default(3),
  }),
});

type ModelConfig = z.infer<typeof modelConfigSchema>;

// Default configuration
const defaultConfig: ModelConfig = {
  modelsDir: path.join(process.cwd(), 'models'),
  cacheDir: path.join(process.cwd(), '.cache'),
  imageModel: {
    name: 'efficientnet-b7',
    version: '1.0.0',
    inputSize: [224, 224],
    batchSize: 32,
  },
  videoModel: {
    name: 'x3d_xs',
    version: '1.0.0',
    frameCount: 16,
    fps: 30,
  },
  audioModel: {
    name: 'wav2vec2-base',
    version: '1.0.0',
    sampleRate: 16000,
    duration: 5,
  },
  performance: {
    useGpu: true,
    maxWorkers: 4,
    maxBatchSize: 8,
    cacheModels: true,
  },
  download: {
    baseUrl: 'https://models.satyaai.tech',
    timeout: 300000,
    retries: 3,
  },
};

// Load and validate configuration
export function loadModelConfig(overrides: Partial<ModelConfig> = {}): ModelConfig {
  try {
    // Merge with environment variables
    const envConfig: Partial<ModelConfig> = {
      performance: {
        useGpu: process.env.USE_GPU !== 'false',
        maxWorkers: process.env.MAX_WORKERS ? parseInt(process.env.MAX_WORKERS, 10) : undefined,
        maxBatchSize: process.env.MAX_BATCH_SIZE ? parseInt(process.env.MAX_BATCH_SIZE, 10) : undefined,
      },
    };

    // Merge configurations with overrides taking precedence
    const mergedConfig = {
      ...defaultConfig,
      ...envConfig,
      ...overrides,
      imageModel: { ...defaultConfig.imageModel, ...overrides.imageModel },
      videoModel: { ...defaultConfig.videoModel, ...overrides.videoModel },
      audioModel: { ...defaultConfig.audioModel, ...overrides.audioModel },
      performance: { ...defaultConfig.performance, ...overrides.performance },
      download: { ...defaultConfig.download, ...overrides.download },
    };

    // Validate the final configuration
    return modelConfigSchema.parse(mergedConfig);
  } catch (error) {
    console.error('Invalid model configuration:', error);
    throw new Error('Failed to load model configuration');
  }
}

// Export default configuration
export const modelConfig = loadModelConfig();

// Helper functions
export function getModelPath(modelName: string, version: string): string {
  return path.join(
    modelConfig.modelsDir,
    `models--${modelName.replace('/', '--')}`, // Hugging Face format
    `refs--${version}`,
    'model.safetensors'
  );
}

export function getCacheKey(modelName: string, version: string): string {
  return `${modelName}@${version}`;
}
