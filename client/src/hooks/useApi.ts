

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api/apiClient';
import { analysisService } from '../lib/api/services/analysisService';

// Types
export interface AnalysisResult {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  is_deepfake: boolean;
  confidence: number;
  model_info: Record<string, unknown>;
  timestamp: string;
  evidence_id: string;
  proof: {
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
  };
  file_name: string;
  file_size: number;
  modality: 'image' | 'audio' | 'video' | 'multimodal';
}

export interface HistoryItem {
  id: string;
  file_name: string;
  modality: string;
  created_at: string;
  confidence: number;
  is_deepfake: boolean;
  status: string;
}

export interface DashboardStats {
  total_analyses: number;
  deepfake_detected: number;
  real_detected: number;
  avg_confidence: number;
  last_7_days: Array<{
    date: string;
    count: number;
  }>;
}

export interface UserAnalytics {
  usage_by_type: {
    image: number;
    audio: number;
    video: number;
    multimodal: number;
    webcam: number;
  };
  recent_activity: Array<{
    id: string;
    modality: string;
    created_at: string;
    confidence: number;
    is_deepfake: boolean;
    file_name: string;
  }>;
}

// Analysis Hooks
export const useImageAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeImage = useMutation({
    mutationFn: async ({ file, options }: { file: File; options?: Record<string, unknown> }) => {
      const response = await analysisService.analyzeImage(file, options);
      return response;
    },
    onSuccess: () => {
      // Invalidate history and stats queries
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    }
  });

  return {
    analyzeImage: analyzeImage.mutateAsync,
    isAnalyzing: analyzeImage.isPending,
    error: analyzeImage.error,
    data: analyzeImage.data
  };
};

export const useAudioAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeAudio = useMutation({
    mutationFn: async ({ file, options }: { file: File; options?: Record<string, unknown> }) => {
      const response = await analysisService.analyzeAudio(file, options);
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    }
  });

  return {
    analyzeAudio: analyzeAudio.mutateAsync,
    isAnalyzing: analyzeAudio.isPending,
    error: analyzeAudio.error,
    data: analyzeAudio.data
  };
};

export const useVideoAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeVideo = useMutation({
    mutationFn: async ({ file, options }: { file: File; options?: Record<string, unknown> }) => {
      const response = await analysisService.analyzeVideo(file, options);
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    }
  });

  return {
    analyzeVideo: analyzeVideo.mutateAsync,
    isAnalyzing: analyzeVideo.isPending,
    error: analyzeVideo.error,
    data: analyzeVideo.data
  };
};

export const useMultimodalAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeMultimodal = useMutation({
    mutationFn: async ({ files }: { files: File[] }) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append(`files`, file);
      });
      
      const response = await apiClient.post('/analysis/multimodal', {
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    }
  });

  return {
    analyzeMultimodal: analyzeMultimodal.mutateAsync,
    isAnalyzing: analyzeMultimodal.isPending,
    error: analyzeMultimodal.error,
    data: analyzeMultimodal.data
  };
};

// History Hooks
export const useAnalysisHistory = (options?: { limit?: number }) => {
  const queryClient = useQueryClient();
  
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['analysis-history', options?.limit],
    queryFn: async () => {
      const response = await apiClient.get('/history', {
        params: options
      });
      return response as { data: { jobs: HistoryItem[] } };
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  });

  const clearHistory = async () => {
    await apiClient.delete('/history');
    queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
    queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
  };

  const deleteHistoryItem = async (jobId: string) => {
    await apiClient.delete(`/history/${jobId}`);
    queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
    queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
  };

  return {
    history: data?.data?.jobs || [],
    isLoading,
    error,
    refetch,
    clearHistory,
    deleteHistoryItem
  };
};

// Dashboard Hooks
export const useDashboardStats = () => {
  return useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await apiClient.get('/dashboard/stats');
      return response as DashboardStats;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 30 * 1000, // 30 seconds
    placeholderData: {
      total_analyses: 0,
      deepfake_detected: 0,
      real_detected: 0,
      avg_confidence: 0,
      last_7_days: []
    }
  });
};

export const useUserAnalytics = () => {
  return useQuery({
    queryKey: ['user-analytics'],
    queryFn: async () => {
      const response = await apiClient.get('/user/analytics');
      return response as UserAnalytics;
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Utility hook for batch operations
export const useBatchAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeBatch = useMutation({
    mutationFn: async ({ files }: { files: File[] }) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append(`files`, file);
      });
      
      const response = await apiClient.post('/analysis/batch', {
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    }
  });

  return {
    analyzeBatch: analyzeBatch.mutateAsync,
    isAnalyzing: analyzeBatch.isPending,
    error: analyzeBatch.error,
    data: analyzeBatch.data
  };
};