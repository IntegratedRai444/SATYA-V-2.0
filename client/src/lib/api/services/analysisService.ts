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
  job_id: string; // Updated to match unified endpoint
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
  
  // File size limits (in bytes)
  private readonly FILE_LIMITS = {
    image: 50 * 1024 * 1024, // 50MB
    video: 500 * 1024 * 1024, // 500MB
    audio: 100 * 1024 * 1024, // 100MB
    text: 10 * 1024 * 1024, // 10MB
  };

  constructor() {
    super(''); // Use empty base path since we're using direct routes
    this.setupOfflineSync();
  }

  async analyzeImage(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);

    if (options.metadata) {
      formData.append('metadata', JSON.stringify(options.metadata));
    }

    try {
      const response = await this.post<JobStartResponse>('analysis/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      // Contract validation logging
      console.log('[API RESPONSE] Raw backend response:', response);
      console.log('[CONTRACT CHECK] Response structure:', {
        success: response?.success,
        job_id: response?.job_id,
        status: response?.status,
        hasJobId: !!response?.job_id,
        hasStatus: !!response?.status
      });

      if (!response.success || !response.job_id || !response.status) {
        console.log('[CONTRACT CHECK] FAILED - Missing required fields:', {
          hasSuccess: !!response?.success,
          hasJobId: !!response?.job_id,
          hasStatus: !!response?.status,
          successValue: response?.success,
          jobIdValue: response?.job_id,
          statusValue: response?.status
        });
        throw new Error('Failed to start analysis job');
      }

      console.log('[FRONTEND VALIDATION] Parsed response successfully, jobId:', response.job_id, 'status:', response.status);
      return response.job_id;
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
      const response = await this.post<JobStartResponse>('analysis/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      // Contract validation logging
      console.log('[API RESPONSE] Raw backend response (video):', response);
      console.log('[CONTRACT CHECK] Response structure (video):', {
        success: response?.success,
        job_id: response?.job_id,
        status: response?.status,
        hasJobId: !!response?.job_id,
        hasStatus: !!response?.status
      });

      if (!response.success || !response.job_id || !response.status) {
        console.log('[CONTRACT CHECK] FAILED - Missing required fields (video):', {
          hasSuccess: !!response?.success,
          hasJobId: !!response?.job_id,
          hasStatus: !!response?.status,
          successValue: response?.success,
          jobIdValue: response?.job_id,
          statusValue: response?.status
        });
        throw new Error('Failed to start analysis job');
      }

      console.log('[FRONTEND VALIDATION] Parsed response successfully, jobId (video):', response.job_id, 'status:', response.status);
      return response.job_id;
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
      const response = await this.post<JobStartResponse>('analysis/audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      // Contract validation logging
      console.log('[API RESPONSE] Raw backend response (audio):', response);
      console.log('[CONTRACT CHECK] Response structure (audio):', {
        success: response?.success,
        job_id: response?.job_id,
        status: response?.status,
        hasJobId: !!response?.job_id,
        hasStatus: !!response?.status
      });

      if (!response.success || !response.job_id || !response.status) {
        console.log('[CONTRACT CHECK] FAILED - Missing required fields (audio):', {
          hasSuccess: !!response?.success,
          hasJobId: !!response?.job_id,
          hasStatus: !!response?.status,
          successValue: response?.success,
          jobIdValue: response?.job_id,
          statusValue: response?.status
        });
        throw new Error('Failed to start analysis job');
      }

      console.log('[FRONTEND VALIDATION] Parsed response successfully, jobId (audio):', response.job_id, 'status:', response.status);
      return response.job_id;
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
   * Check rate limiting
   */
  private checkRateLimit(userId: string): boolean {
    const now = Date.now();
    const userLimit = this.rateLimits.get(userId);
    
    if (!userLimit || now > userLimit.resetTime) {
      // Reset or create new limit
      this.rateLimits.set(userId, {
        count: 1,
        lastRequest: now,
        resetTime: now + this.RATE_WINDOW
      });
      return true;
    }
    
    if (userLimit.count >= this.RATE_LIMIT) {
      return false;
    }
    
    userLimit.count++;
    userLimit.lastRequest = now;
    return true;
  }

  /**
   * Setup offline synchronization
   */
  private setupOfflineSync(): void {
    // Check for online status
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => {
        this.processOfflineQueue();
      });
      
      // Load queued items from localStorage
      try {
        const queued = localStorage.getItem('satya_analysis_queue');
        if (queued) {
          const items = JSON.parse(queued);
          items.forEach((item: QueuedAnalysis) => {
            this.offlineQueue.set(item.id, item);
          });
        }
      } catch (error) {
        console.warn('Failed to load offline queue:', error);
      }
    }
  }

  /**
   * Process offline queue when back online
   */
  private async processOfflineQueue(): Promise<void> {
    for (const [id, queued] of this.offlineQueue) {
      try {
        // Retry the analysis
        if (queued.retryCount < 3) {
          queued.retryCount++;
          await this.retryAnalysis(queued);
          this.offlineQueue.delete(id);
        } else {
          // Max retries reached, remove from queue
          this.offlineQueue.delete(id);
        }
      } catch (error) {
        console.warn(`Failed to retry queued analysis ${id}:`, error);
      }
    }
    this.saveOfflineQueue();
  }

  /**
   * Save offline queue to localStorage
   */
  private saveOfflineQueue(): void {
    if (typeof window !== 'undefined') {
      try {
        const items = Array.from(this.offlineQueue.values());
        localStorage.setItem('satya_analysis_queue', JSON.stringify(items));
      } catch (error) {
        console.warn('Failed to save offline queue:', error);
      }
    }
  }

  /**
   * Retry a failed analysis
   */
  private async retryAnalysis(queued: QueuedAnalysis): Promise<string> {
    // Determine analysis type based on file
    const type = this.getFileType(queued.file);
    
    switch (type) {
      case 'image':
        return this.analyzeImage(queued.file, queued.options);
      case 'video':
        return this.analyzeVideo(queued.file, queued.options);
      case 'audio':
        return this.analyzeAudio(queued.file, queued.options);
      default:
        throw new Error(`Unsupported file type: ${type}`);
    }
  }

  /**
   * Get file type from file
   */
  private getFileType(file: File): 'image' | 'video' | 'audio' | 'text' {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('video/')) return 'video';
    if (file.type.startsWith('audio/')) return 'audio';
    if (file.type.startsWith('text/') || file.type === 'application/json') return 'text';
    return 'image'; // Default fallback
  }

  /**
   * Enhanced analyzeImage with all improvements
   */
  async analyzeImageEnhanced(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    // 1. Validate file first
    const validation = await this.validateFile(file, 'image');
    if (!validation.isValid) {
      throw new Error(`File validation failed: ${validation.errors.join(', ')}`);
    }
    
    // 2. Check rate limiting (mock user ID for now)
    const userId = 'current_user'; // In real app, get from auth context
    if (!this.checkRateLimit(userId)) {
      throw new Error('Rate limit exceeded. Please wait before making another request.');
    }
    
    // 3. Set up progress tracking
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const controller = new AbortController();
    this.activeConnections.set(jobId, controller);
    
    // 4. Notify status change
    options.onStatusChange?.('uploading');
    options.onProgress?.(0, 'Starting upload...');
    
    try {
      // 5. Check and refresh token if needed
      await this.ensureValidToken();
      
      // 6. Perform the analysis with progress tracking
      const result = await this.analyzeImageWithProgress(file, options, controller);
      
      // 7. Clean up
      this.activeConnections.delete(jobId);
      options.onStatusChange?.('completed');
      options.onProgress?.(100, 'Analysis completed');
      
      return result;
    } catch (error) {
      // 8. Handle errors with retry logic
      this.activeConnections.delete(jobId);
      options.onStatusChange?.('failed');
      
      // Add to offline queue if network error
      if (error instanceof Error && 
          (error.message.includes('Network Error') || error.message.includes('ERR_CONNECTION_REFUSED'))) {
        this.addToOfflineQueue(jobId, file, options);
        throw new Error('Network error. Analysis has been queued for retry when connection is restored.');
      }
      
      throw error;
    }
  }

  /**
   * Ensure we have a valid token
   */
  private async ensureValidToken(): Promise<void> {
    // For now, we'll assume the BaseService handles token validation
    // In a real implementation, you would get the token from your auth context
    console.log('Token validation would happen here');
  }

  /**
   * Analyze image with progress tracking
   */
  private async analyzeImageWithProgress(
    file: File, 
    options: AnalysisOptions, 
    controller: AbortController
  ): Promise<string> {
    // Simulate progress updates (in real app, use WebSocket)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 10;
      if (progress > 90) progress = 90;
      options.onProgress?.(Math.round(progress), 'Processing...');
    }, 1000);
    
    try {
      const result = await this.analyzeImage(file, {
        ...options,
        signal: controller.signal
      });
      
      clearInterval(progressInterval);
      options.onProgress?.(100, 'Completed');
      return result;
    } catch (error) {
      clearInterval(progressInterval);
      throw error;
    }
  }

  /**
   * Add analysis to offline queue
   */
  private addToOfflineQueue(id: string, file: File, options: AnalysisOptions): void {
    const queued: QueuedAnalysis = {
      id,
      file,
      options,
      timestamp: Date.now(),
      retryCount: 0
    };
    
    this.offlineQueue.set(id, queued);
    this.saveOfflineQueue();
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