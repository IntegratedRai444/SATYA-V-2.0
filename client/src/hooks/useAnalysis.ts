import { useState, useCallback } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import apiClient from '../lib/api';
import { useWebSocket } from './useWebSocket';

interface AnalysisResult {
  success: boolean;
  result?: {
    authenticity: 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';
    confidence: number;
    analysisDate: string;
    caseId: string;
    keyFindings: string[];
    metrics?: {
      processingTime: number;
      facesDetected?: number;
      framesAnalyzed?: number;
      audioSegments?: number;
    };
    details?: {
      modelVersion: string;
      analysisMethod: string;
      confidenceBreakdown?: Record<string, number>;
      technicalDetails?: Record<string, any>;
    };
    fileInfo?: {
      originalName: string;
      size: number;
      mimeType: string;
    };
  };
  jobId?: string;
  async?: boolean;
  estimatedTime?: number;
  error?: string;
}

interface AnalysisProgress {
  fileId: string;
  filename: string;
  jobId?: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error' | 'queued';
  message?: string;
  result?: any;
}

interface AnalysisOptions {
  sensitivity?: 'low' | 'medium' | 'high';
  includeDetails?: boolean;
  async?: boolean;
}

const analyzeFile = async ({ 
  file, 
  type, 
  options 
}: { 
  file: File; 
  type: 'image' | 'video' | 'audio'; 
  options?: AnalysisOptions;
}): Promise<AnalysisResult> => {
  switch (type) {
    case 'image':
      return await apiClient.analyzeImage(file, options);
    case 'video':
      return await apiClient.analyzeVideo(file, options);
    case 'audio':
      return await apiClient.analyzeAudio(file, options);
    default:
      throw new Error(`Unsupported file type: ${type}`);
  }
};

export const useAnalysis = () => {
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress[]>([]);
  const queryClient = useQueryClient();
  const { subscribeToJob, unsubscribeFromJob } = useWebSocket();

  const analysisMutation = useMutation({
    mutationFn: analyzeFile,
    onSuccess: (result, variables) => {
      const fileId = `${variables.file.name}-${Date.now()}`;
      
      if (result.success) {
        if (result.async && result.jobId) {
          // Async analysis - update with job ID and subscribe to progress
          setAnalysisProgress(prev => 
            prev.map(p => 
              p.fileId === fileId
                ? { 
                    ...p, 
                    jobId: result.jobId, 
                    status: 'queued' as const,
                    message: 'Queued for analysis'
                  }
                : p
            )
          );
          
          // Subscribe to job progress
          subscribeToJob(result.jobId);
        } else {
          // Sync analysis - completed immediately
          setAnalysisProgress(prev => 
            prev.map(p => 
              p.fileId === fileId
                ? { 
                    ...p, 
                    progress: 100, 
                    status: 'completed' as const,
                    result: result.result
                  }
                : p
            )
          );
        }
      } else {
        // Analysis failed
        setAnalysisProgress(prev => 
          prev.map(p => 
            p.fileId === fileId
              ? { 
                  ...p, 
                  status: 'error' as const, 
                  message: result.error || 'Analysis failed' 
                }
              : p
          )
        );
      }
      
      // Invalidate and refetch analysis results
      queryClient.invalidateQueries({ queryKey: ['analysis-results'] });
    },
    onError: (error, variables) => {
      const fileId = `${variables.file.name}-${Date.now()}`;
      
      setAnalysisProgress(prev => 
        prev.map(p => 
          p.fileId === fileId
            ? { 
                ...p, 
                status: 'error' as const, 
                message: (error as Error).message || 'Analysis failed' 
              }
            : p
        )
      );
    }
  });

  const startAnalysis = useCallback(async (
    files: File[], 
    type: 'image' | 'video' | 'audio',
    options?: AnalysisOptions
  ) => {
    // Initialize progress for all files
    const newProgress: AnalysisProgress[] = files.map(file => ({
      fileId: `${file.name}-${Date.now()}`,
      filename: file.name,
      progress: 0,
      status: 'uploading' as const
    }));

    setAnalysisProgress(prev => [...prev, ...newProgress]);

    // Process each file
    for (const file of files) {
      try {
        // Update to processing
        setAnalysisProgress(prev => 
          prev.map(p => 
            p.filename === file.name 
              ? { ...p, progress: 25, status: 'processing' as const }
              : p
          )
        );

        // Start actual analysis
        await analysisMutation.mutateAsync({ file, type, options });
      } catch (error) {
        console.error('Analysis failed for file:', file.name, error);
      }
    }
  }, [analysisMutation, subscribeToJob]);

  const startMultimodalAnalysis = useCallback(async (
    files: { image?: File; video?: File; audio?: File },
    options?: AnalysisOptions
  ) => {
    const fileId = `multimodal-${Date.now()}`;
    const fileNames = [
      files.image?.name,
      files.video?.name,
      files.audio?.name
    ].filter(Boolean).join(', ');

    // Initialize progress
    const newProgress: AnalysisProgress = {
      fileId,
      filename: `Multimodal: ${fileNames}`,
      progress: 0,
      status: 'uploading'
    };

    setAnalysisProgress(prev => [...prev, newProgress]);

    try {
      setAnalysisProgress(prev => 
        prev.map(p => 
          p.fileId === fileId
            ? { ...p, progress: 25, status: 'processing' as const }
            : p
        )
      );

      const result = await apiClient.analyzeMultimodal(files, options);

      if (result.success) {
        if (result.async && result.jobId) {
          setAnalysisProgress(prev => 
            prev.map(p => 
              p.fileId === fileId
                ? { 
                    ...p, 
                    jobId: result.jobId, 
                    status: 'queued' as const,
                    message: 'Queued for analysis'
                  }
                : p
            )
          );
          
          subscribeToJob(result.jobId);
        } else {
          setAnalysisProgress(prev => 
            prev.map(p => 
              p.fileId === fileId
                ? { 
                    ...p, 
                    progress: 100, 
                    status: 'completed' as const,
                    result: result.result
                  }
                : p
            )
          );
        }
      } else {
        setAnalysisProgress(prev => 
          prev.map(p => 
            p.fileId === fileId
              ? { 
                  ...p, 
                  status: 'error' as const, 
                  message: result.error || 'Analysis failed' 
                }
              : p
          )
        );
      }

      queryClient.invalidateQueries({ queryKey: ['analysis-results'] });
    } catch (error) {
      setAnalysisProgress(prev => 
        prev.map(p => 
          p.fileId === fileId
            ? { 
                ...p, 
                status: 'error' as const, 
                message: (error as Error).message || 'Analysis failed' 
              }
            : p
        )
      );
    }
  }, [subscribeToJob, queryClient]);

  const clearProgress = useCallback(() => {
    // Unsubscribe from all jobs
    analysisProgress.forEach(progress => {
      if (progress.jobId) {
        unsubscribeFromJob(progress.jobId);
      }
    });
    
    setAnalysisProgress([]);
  }, [analysisProgress, unsubscribeFromJob]);

  const removeProgress = useCallback((fileId: string) => {
    const progress = analysisProgress.find(p => p.fileId === fileId);
    if (progress?.jobId) {
      unsubscribeFromJob(progress.jobId);
    }
    
    setAnalysisProgress(prev => prev.filter(p => p.fileId !== fileId));
  }, [analysisProgress, unsubscribeFromJob]);

  // Query for analysis history
  const { data: analysisHistory, isLoading: isLoadingHistory } = useQuery({
    queryKey: ['analysis-history'],
    queryFn: () => apiClient.getAnalysisHistory(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Query for dashboard stats
  const { data: dashboardStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: () => apiClient.getDashboardStats(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });

  return {
    startAnalysis,
    startMultimodalAnalysis,
    analysisProgress,
    clearProgress,
    removeProgress,
    isAnalyzing: analysisMutation.isPending,
    analysisHistory: analysisHistory?.data || [],
    isLoadingHistory,
    dashboardStats: dashboardStats?.data,
    isLoadingStats
  };
};