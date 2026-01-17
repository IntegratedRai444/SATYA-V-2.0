import { useState, useCallback } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { analysisService, type AnalysisResult as ServiceAnalysisResult, type AnalysisOptions as ServiceAnalysisOptions } from '../lib/api';
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
  const serviceOptions = options as ServiceAnalysisOptions;
  
  switch (type) {
    case 'image':
      const imageResult = await analysisService.analyzeImage(file, serviceOptions);
      return {
        success: imageResult.status === 'completed',
        result: imageResult.result,
        error: imageResult.error
      };
    case 'video':
      const videoResult = await analysisService.analyzeVideo(file, serviceOptions);
      return {
        success: videoResult.status === 'completed',
        result: videoResult.result,
        error: videoResult.error
      };
    case 'audio':
      const audioResult = await analysisService.analyzeAudio(file, serviceOptions);
      return {
        success: audioResult.status === 'completed',
        result: audioResult.result,
        error: audioResult.error
      };
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

      let result: AnalysisResult;
      const serviceOptions = options as ServiceAnalysisOptions;
      
      if (files.image) {
        const imageResult = await analysisService.analyzeImage(files.image, serviceOptions);
        result = {
          success: imageResult.status === 'completed',
          result: imageResult.result,
          error: imageResult.error
        };
      } else if (files.video) {
        const videoResult = await analysisService.analyzeVideo(files.video, serviceOptions);
        result = {
          success: videoResult.status === 'completed',
          result: videoResult.result,
          error: videoResult.error
        };
      } else if (files.audio) {
        const audioResult = await analysisService.analyzeAudio(files.audio, serviceOptions);
        result = {
          success: audioResult.status === 'completed',
          result: audioResult.result,
          error: audioResult.error
        };
      } else {
        throw new Error('No files provided for multimodal analysis');
      }

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
