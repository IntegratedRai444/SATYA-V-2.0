

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import { analysisService } from '../lib/api/services/analysisService';

// Types
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
  fileName: string;
  fileSize: number;
  userId?: string;
  requestId?: string;
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

export interface HistoryItem {
  id: string;
  fileName: string;
  modality: string;
  createdAt: string;
  confidence: number;
  isAuthentic: boolean;
  status: string;
}

export interface DashboardStats {
  totalAnalyses: number;
  deepfakeDetected: number;
  realDetected: number;
  avgConfidence: number;
  last7Days: Array<{
    date: string;
    count: number;
    deepfakes: number;
  }>;
}

export interface UserAnalytics {
  usageByType: {
    image: number;
    audio: number;
    video: number;
    multimodal: number;
  };
  recentActivity: Array<{
    id: string;
    modality: string;
    createdAt: string;
    confidence: number;
    isAuthentic: boolean;
    fileName: string;
  }>;
}

// Analysis Hooks
export const useImageAnalysis = () => {
  const queryClient = useQueryClient();
  
  const analyzeImage = useMutation({
    mutationFn: async ({ file, options }: { file: File; options?: Record<string, unknown> }) => {
      const jobId = await analysisService.analyzeImage(file, options);
      return { jobId }; // Return object with jobId
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
      const jobId = await analysisService.analyzeAudio(file, options);
      return { jobId }; // Return object with jobId
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
      const jobId = await analysisService.analyzeVideo(file, options);
      return { jobId }; // Return object with jobId
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
      
      const jobId = await analysisService.analyzeMultimodal(files[0], { metadata: { files } });
      return { jobId }; // Return object with jobId
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
      return response.data as DashboardStats;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 30 * 1000, // 30 seconds
    placeholderData: {
      totalAnalyses: 0,
      deepfakeDetected: 0,
      realDetected: 0,
      avgConfidence: 0,
      last7Days: []
    }
  });
};

export const useUserAnalytics = () => {
  return useQuery({
    queryKey: ['user-analytics'],
    queryFn: async () => {
      const response = await apiClient.get('/user/analytics');
      return response.data as UserAnalytics;
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