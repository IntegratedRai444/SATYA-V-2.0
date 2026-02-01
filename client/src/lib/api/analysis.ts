import { apiClient } from './client';

// Define types for analysis history
export interface AnalysisResult {
  id: string;
  type: 'image' | 'video' | 'audio';
  filename: string;
  result: {
    confidence: number;
    is_deepfake: boolean;
    model_name?: string;
    model_version?: string;
    summary?: Record<string, unknown>;
  };
  timestamp: string;
  isDeepfake: boolean;
  confidence: number;
  status: 'completed' | 'processing' | 'failed';
}

// Get analysis history from the server
export const getAnalysisHistory = async (): Promise<AnalysisResult[]> => {
  try {
    const response = await apiClient.get('/history');
    return response.data.data?.jobs || [];
  } catch (error) {
    console.error('Error fetching analysis history:', error);
    throw error;
  }
};

// Clear analysis history
export const clearAnalysisHistory = async (): Promise<void> => {
  try {
    await apiClient.delete('/history');
  } catch (error) {
    console.error('Error clearing analysis history:', error);
    throw error;
  }
};

// Get analysis result by ID
export const getAnalysisResult = async (id: string): Promise<AnalysisResult> => {
  try {
    const response = await apiClient.get(`/results/${id}`);
    return response.data.data;
  } catch (error) {
    console.error(`Error fetching analysis result ${id}:`, error);
    throw error;
  }
};

// Submit new analysis
export const submitAnalysis = async (file: File, type: 'image' | 'video' | 'audio'): Promise<{ id: string }> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    
    const response = await apiClient.post(`/analysis/${type}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data.data;
  } catch (error) {
    console.error('Error submitting analysis:', error);
    throw error;
  }
};
