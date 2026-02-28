/**
 * AI Engine Warmup Service
 * Initializes Python ML components during Node.js startup to prevent cold starts
 */

import { pythonBridge } from '../services/python-http-bridge';
import { logger } from '../config/logger';

interface WarmupResult {
  success: boolean;
  warmupTime: number;
  modelsLoaded: boolean;
  memoryUsage: number;
  error?: string;
  details?: Record<string, unknown>;
}

export class AIEngineWarmup {
  private static instance: AIEngineWarmup;
  private warmupCompleted = false;
  private warmupInProgress = false;
  private lastWarmupTime = 0;
  private warmupInterval = 5 * 60 * 1000; // 5 minutes

  static getInstance(): AIEngineWarmup {
    if (!AIEngineWarmup.instance) {
      AIEngineWarmup.instance = new AIEngineWarmup();
    }
    return AIEngineWarmup.instance;
  }

  /**
   * Perform AI engine warmup during Node.js startup
   */
  async performWarmup(): Promise<WarmupResult> {
    if (this.warmupCompleted && !this.shouldWarmup()) {
      logger.info('[AI_WARMUP] Already completed, skipping');
      return {
        success: true,
        warmupTime: 0,
        modelsLoaded: true,
        memoryUsage: 0
      };
    }

    if (this.warmupInProgress) {
      logger.warn('[AI_WARMUP] Warmup already in progress, skipping');
      return {
        success: false,
        warmupTime: 0,
        modelsLoaded: false,
        memoryUsage: 0,
        error: 'Warmup already in progress'
      };
    }

    this.warmupInProgress = true;
    const warmupStart = Date.now();

    try {
      logger.info('[AI_WARMUP] Starting AI engine warmup...');

      // Call Python warmup endpoint
      const response = await pythonBridge.get('/warmup', {
        timeout: 30000, // 30 seconds for warmup
      });

      const warmupTime = Date.now() - warmupStart;
      const results = response as {
        warmup: string;
        results?: {
          models_loaded?: boolean;
          memory_usage?: number;
        };
      };

      if (results?.warmup === 'completed') {
        this.warmupCompleted = true;
        this.lastWarmupTime = Date.now();

        logger.info('[AI_WARMUP] AI engine warmup completed successfully', {
          warmupTime,
          modelsLoaded: results.results?.models_loaded || false,
          memoryUsage: results.results?.memory_usage || 0
        });

        return {
          success: true,
          warmupTime,
          modelsLoaded: results.results?.models_loaded || false,
          memoryUsage: results.results?.memory_usage || 0,
          details: results.results
        };
      } else {
        throw new Error('Warmup failed - no completion response');
      }

    } catch (error) {
      const warmupTime = Date.now() - warmupStart;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      logger.error('[AI_WARMUP] AI engine warmup failed', {
        warmupTime,
        error: errorMessage
      });

      return {
        success: false,
        warmupTime,
        modelsLoaded: false,
        memoryUsage: 0,
        error: errorMessage
      };

    } finally {
      this.warmupInProgress = false;
    }
  }

  /**
   * Check if warmup should be performed
   */
  private shouldWarmup(): boolean {
    return Date.now() - this.lastWarmupTime > this.warmupInterval;
  }

  /**
   * Get warmup status
   */
  getStatus(): {
    completed: boolean;
    inProgress: boolean;
    lastWarmup: number;
  } {
    return {
      completed: this.warmupCompleted,
      inProgress: this.warmupInProgress,
      lastWarmup: this.lastWarmupTime
    };
  }

  /**
   * Reset warmup state (for testing or manual retry)
   */
  reset(): void {
    this.warmupCompleted = false;
    this.warmupInProgress = false;
    this.lastWarmupTime = 0;
    logger.info('[AI_WARMUP] Warmup state reset');
  }
}

export default AIEngineWarmup;
