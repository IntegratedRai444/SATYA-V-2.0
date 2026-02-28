import { BaseService } from './baseService';

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  description?: string;
  isActive: boolean;
  supportedTypes: Array<'image' | 'video' | 'audio' | 'text'>;
  createdAt: string;
  updatedAt: string;
}

export interface AnalysisOptions {
  // Core options
  sensitivity?: 'low' | 'medium' | 'high';
  includeDetails?: boolean;
  timeout?: number;
  signal?: AbortSignal;
  
  // Advanced options
  language?: string;
  modelId?: string;
  metadata?: Record<string, unknown>;
  
  // Media-specific options
  text?: string;
  threshold?: number;
  
  // For batch processing
  batchId?: string;
  priority?: 'low' | 'normal' | 'high';
  
  // Progress tracking
  onProgress?: (progress: number, status: string) => void;
  onStatusChange?: (status: 'uploading' | 'processing' | 'completed' | 'failed') => void;
}

export interface AnalysisResult {
  id: string;
  type: 'image' | 'video' | 'audio' | 'text';
  status: 'processing' | 'completed' | 'failed';
  proof?: AnalysisProof;
  result?: {
    isAuthentic: boolean;
    confidence: number;
    details: Record<string, unknown>;
    metrics: {
      processingTime: number;
      modelVersion: string;
    };
  };
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface JobStartResponse {
  success: boolean;
  jobId: string; // Updated to match unified endpoint
  status: string; // Add missing status field
}

export interface AnalysisProof {
  model_name: string;
  model_version: string;
  modality: string;
  timestamp: string;
  inference_duration: number;
  frames_analyzed: number;
  signature: string;
  metadata: {
    request_id: string;
    user_id: string;
    analysis_type: string;
    content_size: number;
  };
}

// File validation interface
export interface FileValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  metadata: {
    type: string;
    size: number;
    mimeType: string;
    isImage: boolean;
    isVideo: boolean;
    isAudio: boolean;
  };
}

// Rate limiting interface
interface RateLimitEntry {
  count: number;
  lastRequest: number;
  resetTime: number;
}

// Offline queue interface
interface QueuedAnalysis {
  id: string;
  file: File;
  options: AnalysisOptions;
  timestamp: number;
  retryCount: number;
}

export class AnalysisService extends BaseService {
  private rateLimits = new Map<string, RateLimitEntry>();
  private offlineQueue = new Map<string, QueuedAnalysis>();
  private activeConnections = new Map<string, AbortController>();
  
  // Rate limiting: 5 requests per minute per user
  private readonly RATE_LIMIT = 5;
  private readonly RATE_WINDOW = 60000; // 1 minute
  private readonly MAX_REQUESTS = 5; // Maximum requests per window
  
  // File size limits (in bytes)
  private readonly FILE_LIMITS = {
    image: 50 * 1024 * 1024, // 50MB
    video: 500 * 1024 * 1024, // 500MB
    audio: 100 * 1024 * 1024, // 100MB
    text: 10 * 1024 * 1024, // 10MB
  };

  constructor() {
    super(''); // Use empty base path since we're using direct routes
    // Note: setupOfflineSync method would be implemented for offline functionality
  }

  /**
   * Check if user is rate limited
   */
  private isRateLimited(userId: string): boolean {
    return !this.checkRateLimit(userId);
  }

  private checkRateLimit(userId: string): boolean {
    const now = Date.now();
    const rateLimitEntry = this.rateLimits.get(userId);

    if (!rateLimitEntry) {
      this.rateLimits.set(userId, {
        count: 1,
        lastRequest: now,
        resetTime: now + this.RATE_WINDOW,
      });
      return true;
    }

    if (rateLimitEntry.resetTime < now) {
      this.rateLimits.set(userId, {
        count: 1,
        lastRequest: now,
        resetTime: now + this.RATE_WINDOW,
      });
      return true;
    }

    if (rateLimitEntry.count < this.MAX_REQUESTS) {
      rateLimitEntry.count++;
      rateLimitEntry.lastRequest = now;
      this.rateLimits.set(userId, rateLimitEntry);
      return true;
    }

    return false;
  }

  async analyzeImage(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    // Check rate limits if userId is provided
    if (options.metadata?.userId) {
      if (this.isRateLimited(options.metadata.userId as string)) {
        throw new Error(`Rate limit exceeded. Maximum ${this.RATE_LIMIT} requests per ${this.RATE_WINDOW / 1000} seconds.`);
      }
    }

    const formData = new FormData();
    formData.append('file', file);

    if (options.metadata) {
      formData.append('metadata', JSON.stringify(options.metadata));
    }

    try {
      const response = await this.post<JobStartResponse>('unified/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 900000, // 15 minutes - MATCH BACKEND TIMEOUT
      });

      return response.jobId;
    } catch (error) {
      console.error('Image analysis failed:', error);
      
      // Provide more specific error messages
      if (error instanceof Error) {
        if (error.message.includes('Network Error') || error.message.includes('ERR_CONNECTION_REFUSED')) {
          throw new Error('Backend services are not running. Please start the Node.js backend on port 5001.');
        }
        if (error.message.includes('timeout')) {
          throw new Error('Analysis request timed out. Please try again with a smaller file.');
        }
        if (error.message.includes('413')) {
          throw new Error('File too large. Please upload a file smaller than 50MB.');
        }
        if (error.message.includes('415')) {
          throw new Error('Unsupported file format. Please upload a valid image file.');
        }
      }
      
      throw new Error('Failed to start image analysis. Please try again.');
    }
  }

  async getAnalysisResult(jobId: string): Promise<AnalysisResult> {
    return this.get('/results/' + jobId);
  }

  async analyzeVideo(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.post<JobStartResponse>('unified/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 900000, // 15 minutes - MATCH BACKEND TIMEOUT
      });

      return response.jobId;
    } catch (error) {
      console.error('Video analysis error:', error);
      throw new Error('Failed to start video analysis');
    }
  }

  async analyzeAudio(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.post<JobStartResponse>('unified/audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 900000, // 15 minutes - MATCH BACKEND TIMEOUT
      });

      return response.jobId;
    } catch (error) {
      console.error('Audio analysis error:', error);
      throw new Error('Failed to start audio analysis');
    }
  }

  // DISABLED: Multimodal analysis temporarily deactivated
  async analyzeMultimodal(
    _file: File,
    _options: AnalysisOptions = {}
  ): Promise<string> {
    throw new Error('Multimodal analysis is temporarily disabled');
  }

  async getAnalysisHistory(params: {
    limit?: number;
    offset?: number;
    type?: string;
  } = {}): Promise<{ items: AnalysisResult[]; total: number }> {
    return this.get<{ items: AnalysisResult[]; total: number }>('/history', params);
  }

  async deleteAnalysis(id: string): Promise<void> {
    await this.delete(`/results/${id}`);
  }

  /**
   * Get a list of available analysis models
   * @param options Filtering and pagination options
   * @returns List of available models with pagination info
   */
  async getModels(options: {
    type?: 'image' | 'video' | 'audio' | 'text';
    activeOnly?: boolean;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ items: ModelInfo[]; total: number }> {
    try {
      return await this.get<{ items: ModelInfo[]; total: number }>('/models', options);
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Return a default model if the API call fails
      return {
        items: [{
          id: 'default-model',
          name: 'Default Model',
          version: '1.0.0',
          description: 'Default analysis model',
          isActive: true,
          supportedTypes: ['image', 'video', 'audio', 'text'],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }],
        total: 1
      };
    }
  }

  // Enhanced methods for the weaknesses we identified
  
  /**
   * Validate file before upload
   */
  async validateFile(file: File, type: 'image' | 'video' | 'audio' | 'text'): Promise<FileValidationResult> {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Check file size
    const limit = this.FILE_LIMITS[type];
    if (file.size > limit) {
      errors.push(`File size ${(file.size / 1024 / 1024).toFixed(1)}MB exceeds limit of ${(limit / 1024 / 1024).toFixed(1)}MB`);
    }
    
    // Check file type
    const validTypes = {
      image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
      video: ['video/mp4', 'video/avi', 'video/mov', 'video/wmv'],
      audio: ['audio/mp3', 'audio/wav', 'audio/m4a', 'audio/ogg'],
      text: ['text/plain', 'text/csv', 'application/json']
    };
    
    if (!validTypes[type].includes(file.type)) {
      errors.push(`Invalid file type ${file.type}. Expected: ${validTypes[type].join(', ')}`);
    }
    
    // Check filename for security
    if (file.name.includes('..') || /[<>:"|?*]/.test(file.name)) {
      errors.push('Filename contains invalid characters');
    }
    
    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      metadata: {
        type,
        size: file.size,
        mimeType: file.type,
        isImage: type === 'image',
        isVideo: type === 'video',
        isAudio: type === 'audio'
      }
    };
  }

  /**
   * Cancel active analysis
   */
  cancelAnalysis(jobId: string): void {
    const controller = this.activeConnections.get(jobId);
    if (controller) {
      controller.abort();
      this.activeConnections.delete(jobId);
    }
  }

  /**
   * Get offline queue status
   */
  getOfflineQueueStatus(): { count: number; items: QueuedAnalysis[] } {
    return {
      count: this.offlineQueue.size,
      items: Array.from(this.offlineQueue.values())
    };
  }
}

export const analysisService = new AnalysisService();

export default analysisService;