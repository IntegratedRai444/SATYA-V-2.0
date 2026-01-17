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
  metadata?: Record<string, any>;
  
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
    details: Record<string, any>;
    metrics: {
      processingTime: number;
      modelVersion: string;
    };
  };
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface AnalysisApiResponse {
  status: string;
  analysis: {
    is_deepfake: boolean;
    confidence: number;
    model_info: Record<string, any>;
    timestamp: string;
    evidence_id: string;
    proof: AnalysisProof;
  };
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

  private validateResponse<T>(response: any): T {
    if (!response) {
      throw new Error('Empty response from server');
    }

    // Check for error response
    if (response.error) {
      throw new Error(response.error.message || 'Analysis failed');
    }

    // Validate proof if present
    if (response.proof) {
      const requiredFields = ['model_name', 'model_version', 'signature', 'timestamp'];
      for (const field of requiredFields) {
        if (response.proof[field] === undefined || response.proof[field] === null) {
          throw new Error(`Invalid proof: missing required field '${field}'`);
        }
      }
    }

    return response as T;
  }


  async analyzeImage(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    // Add request metadata
    if (options.metadata) {
      formData.append('metadata', JSON.stringify(options.metadata));
    }

    try {
      const response = await this.post<AnalysisApiResponse>('/api/analysis/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options.signal,
      });

      // Validate the response including proof
      const validated = this.validateResponse<AnalysisApiResponse>(response);

      if (validated.status !== 'success' || !validated.analysis) {
        throw new Error('Invalid analysis response from server');
      }

      // Verify the proof is valid
      if (!validated.analysis.proof || !validated.analysis.proof.signature) {
        throw new Error('Analysis response is missing valid proof');
      }

      // Verify the proof matches the analysis
      if (validated.analysis.proof.metadata.request_id !== validated.analysis.evidence_id) {
        throw new Error('Proof verification failed: request ID mismatch');
      }

      return {
        id: validated.analysis.evidence_id,
        type: 'image',
        status: 'completed',
        proof: validated.analysis.proof,
        result: {
          isAuthentic: !validated.analysis.is_deepfake,
          confidence: validated.analysis.confidence,
          details: {
            isDeepfake: validated.analysis.is_deepfake,
            modelInfo: validated.analysis.model_info || {},
          },
          metrics: {
            processingTime: validated.analysis.proof.inference_duration || 0,
            modelVersion: validated.analysis.proof.model_version || 'unknown',
          },
        },
        createdAt: validated.analysis.timestamp || new Date().toISOString(),
        updatedAt: validated.analysis.timestamp || new Date().toISOString(),
      };
    } catch (error) {
      console.error('Image analysis failed:', error);
      throw new Error('Failed to analyze image. Please try again.');
    }
  }

  async analyzeVideo(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    return this.upload<AnalysisResult>(
      '/video',
      formData,
      options
    );
  }

  async analyzeAudio(
    file: File,
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    return this.upload<AnalysisResult>(
      '/audio',
      formData,
      options
    );
  }

  async getAnalysisResult(id: string): Promise<AnalysisResult> {
    return this.get<AnalysisResult>(`/results/${id}`);
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

  // Helper method for file uploads with progress tracking
  private async upload<T>(
    endpoint: string,
    data: FormData,
    options: AnalysisOptions = {}
  ): Promise<T> {
    const controller = new AbortController();
    
    // Set up progress tracking if needed
    if (options.metadata?.onUploadProgress) {
      options.metadata.onUploadProgress = (progressEvent: ProgressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / (progressEvent.total || 1)
        );
        (options.metadata as any).onUploadProgress(percentCompleted);
      };
    }

    // Set up timeout if specified
    if (options.timeout) {
      setTimeout(() => {
        controller.abort();
      }, options.timeout);
    }

    // Make the request
    try {
      const response = await this.post<T>(
        endpoint,
        data,
        {
          signal: options.signal || controller.signal,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return response;
    } catch (error: unknown) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request was aborted due to timeout');
        }
      }
      throw error;
    }
  }
}

export const analysisService = new AnalysisService();

export default analysisService;
