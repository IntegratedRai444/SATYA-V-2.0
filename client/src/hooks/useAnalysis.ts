import { useState, useCallback } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { analysisService, type AnalysisResult as ServiceAnalysisResult } from '../lib/api';
import { useWebSocket } from './useWebSocket';

interface AnalysisResult {
  success: boolean;
  result?: ServiceAnalysisResult['result'];
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
  result?: unknown;
}

interface AnalysisOptions {
  sensitivity?: 'low' | 'medium' | 'high';
  includeDetails?: boolean;
  async?: boolean;
  signal?: AbortSignal;
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
  try {
    let jobId: string;
    
    switch (type) {
      case 'image': {
        jobId = await analysisService.analyzeImage(file, options);
        break;
      }
      case 'video': {
        jobId = await analysisService.analyzeVideo(file, options);
        break;
      }
      case 'audio': {
        jobId = await analysisService.analyzeAudio(file, options);
        break;
      }
      default:
        throw new Error(`Unsupported file type: ${type}`);
    }

    return {
      success: true,
      async: true,
      jobId,
      estimatedTime: 30000 // 30 seconds default
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Analysis failed'
    };
  }
};

export const useAnalysis = () => {
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress[]>([]);
  const queryClient = useQueryClient();
  const { subscribeToJob, unsubscribeFromJob, isConnected } = useWebSocket();

  const analysisMutation = useMutation({
    mutationFn: analyzeFile,
    onSuccess: (result, variables) => {
      const fileId = `${variables.file.name}-${Date.now()}`;
      
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
          
          if (isConnected) {
            subscribeToJob(result.jobId);
          }
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
    const newProgress: AnalysisProgress[] = files.map(file => ({
      fileId: `${file.name}-${Date.now()}`,
      filename: file.name,
      progress: 0,
      status: 'uploading' as const
    }));

    setAnalysisProgress(prev => [...prev, ...newProgress]);

    for (const file of files) {
      try {
        setAnalysisProgress(prev => 
          prev.map(p => 
            p.filename === file.name 
              ? { ...p, progress: 25, status: 'processing' as const }
              : p
          )
        );

        await analysisMutation.mutateAsync({ file, type, options });
      } catch (error) {
        console.error('Analysis failed for file:', file.name, error);
      }
    }
  }, [analysisMutation]);

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

      let jobId: string;
      
      if (files.image) {
        jobId = await analysisService.analyzeImage(files.image, options);
      } else if (files.video) {
        jobId = await analysisService.analyzeVideo(files.video, options);
      } else if (files.audio) {
        jobId = await analysisService.analyzeAudio(files.audio, options);
      } else {
        throw new Error('No files provided for multimodal analysis');
      }

      const result: AnalysisResult = {
        success: true,
        async: true,
        jobId,
        estimatedTime: 45000 // 45 seconds for multimodal
      };

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
          
          if (isConnected) {
            subscribeToJob(result.jobId);
          }
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
  }, [subscribeToJob, queryClient, isConnected]);

  const clearProgress = useCallback(() => {
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

  const analysisHistory = useQuery({
    queryKey: ['analysis-history'],
    queryFn: async () => [],
    staleTime: 5 * 60 * 1000,
  });

  const isLoadingHistory = analysisHistory.isLoading;

  const dashboardStats = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => ({}),
    staleTime: 5 * 60 * 1000,
  });

  const isLoadingStats = dashboardStats.isLoading;

  return {
    analysisProgress,
    startAnalysis,
    startMultimodalAnalysis,
    clearProgress,
    removeProgress,
    isAnalyzing: analysisMutation.isPending,
    analysisHistory: analysisHistory?.data || [],
    isLoadingHistory,
    dashboardStats: dashboardStats?.data,
    isLoadingStats
  };
};
