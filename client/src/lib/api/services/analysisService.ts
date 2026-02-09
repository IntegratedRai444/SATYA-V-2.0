import { BaseService } from './baseService';

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  description?: string;
  isActive: boolean;
  supportedTypes: Array<'image' | 'video' | 'audio' | 'text' | 'multimodal'>;
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
}

export interface AnalysisResult {
  id: string;
  type: 'image' | 'video' | 'audio' | 'multimodal';
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

export class AnalysisService extends BaseService {
  constructor() {
    super(''); // Use empty base path since we're using direct routes
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
      const response = await this.post<JobStartResponse>('/api/v2/analysis/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      if (!response.success || !response.job_id) {
        throw new Error('Failed to start analysis job');
      }

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
    return this.get(`/results/${jobId}`);
  }

  async analyzeVideo(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.post<JobStartResponse>('/api/v2/analysis/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      if (!response.success || !response.job_id) {
        throw new Error('Failed to start analysis job');
      }

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
      const response = await this.post<JobStartResponse>('/api/v2/analysis/audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      if (!response.success || !response.job_id) {
        throw new Error('Failed to start analysis job');
      }

      return response.job_id;
    } catch (error) {
      console.error('Audio analysis error:', error);
      throw new Error('Failed to start audio analysis');
    }
  }

  async analyzeMultimodal(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<string> {
    const formData = new FormData();
    formData.append('files', file);

    // Add additional files if provided in metadata
    if (options.metadata?.files) {
      (options.metadata.files as File[]).forEach((f: File) => {
        formData.append('files', f);
      });
    }

    try {
      const response = await this.post<JobStartResponse>('/api/v2/analysis/multimodal', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
        timeout: 360000, // 6 minutes timeout
      });

      if (!response.success || !response.job_id) {
        throw new Error('Failed to start analysis job');
      }

      return response.job_id;
    } catch (error) {
      console.error('Multimodal analysis error:', error);
      throw new Error('Failed to start multimodal analysis');
    }
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
    type?: 'image' | 'video' | 'audio' | 'text' | 'multimodal';
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
          supportedTypes: ['image', 'video', 'audio', 'text', 'multimodal'],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }],
        total: 1
      };
    }
  }
}

export const analysisService = new AnalysisService();

export default analysisService;