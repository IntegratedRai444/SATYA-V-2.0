import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { logger } from '../config/logger';
import { pythonConfig } from '../config';

export interface AnalysisRequest {
  filePath: string;
  fileType: 'image' | 'video' | 'audio';
  userId: string;
  metadata?: Record<string, any>;
}

export interface AnalysisResponse {
  success: boolean;
  result?: {
    isDeepfake: boolean;
    confidence: number;
    modelUsed: string;
    processingTime: number;
    analysisDetails: Record<string, any>;
  };
  error?: string;
}

export class PythonBridge {
  private client: AxiosInstance;
  private isAvailable: boolean = false;

  constructor() {
    this.client = axios.create({
      baseURL: pythonConfig.apiUrl,
      timeout: 60000, // 1 minute timeout
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': pythonConfig.apiKey,
      },
    });

    // Check Python service availability on startup
    this.checkAvailability().catch(err => {
      logger.warn('Python service not available on startup', { error: err.message });
    });
  }

  async checkAvailability(): Promise<boolean> {
    try {
      await this.client.get('/health');
      this.isAvailable = true;
      return true;
    } catch (error) {
      this.isAvailable = false;
      logger.error('Python service check failed', { error });
      return false;
    }
  }

  async analyzeMedia(data: AnalysisRequest): Promise<AnalysisResponse> {
    if (!this.isAvailable) {
      const isAvailable = await this.checkAvailability();
      if (!isAvailable) {
        throw new Error('Python analysis service is not available');
      }
    }

    try {
      const response = await this.client.post<AnalysisResponse>('/analyze', data);
      return response.data;
    } catch (error) {
      const axiosError = error as AxiosError;
      logger.error('Python service request failed', {
        status: axiosError.response?.status,
        message: axiosError.message,
        data: axiosError.response?.data,
      });
      
      this.isAvailable = false; // Mark as unavailable on error
      
      throw new Error(
        axiosError.response?.data?.error || 
        `Analysis failed: ${axiosError.message}`
      );
    }
  }

  async getAnalysisStatus(jobId: string): Promise<AnalysisResponse> {
    try {
      const response = await this.client.get<AnalysisResponse>(`/status/${jobId}`);
      return response.data;
    } catch (error) {
      const axiosError = error as AxiosError;
      throw new Error(
        axiosError.response?.data?.error || 
        `Failed to get analysis status: ${axiosError.message}`
      );
    }
  }

  async shutdown(): Promise<void> {
    try {
      await this.client.post('/shutdown');
      this.isAvailable = false;
    } catch (error) {
      logger.warn('Failed to gracefully shutdown Python service', { error });
    }
  }
}

export const pythonBridge = new PythonBridge();
