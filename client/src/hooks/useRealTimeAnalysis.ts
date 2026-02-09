import { useState, useCallback, useEffect, useRef } from 'react';
import { useJobProgress, JobProgress } from './useWebSocket';
import { useToast } from '@/components/ui/use-toast';
import logger from '../lib/logger';

type TimeoutId = ReturnType<typeof setTimeout>;

interface RealTimeAnalysisOptions {
  jobId?: string | null;
  onProgress?: (progress: JobProgress | unknown) => void;
  onComplete?: (result: unknown) => void;
  onError?: (error: string) => void;
  autoRetry?: boolean;
  maxRetries?: number;
}

interface AnalysisState {
  status: 'idle' | 'connecting' | 'analyzing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  stage: string;
  message: string;
  startTime?: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
  error?: string;
  result?: unknown;
  metrics?: {
    processingTime?: number;
    modelVersion?: string;
    confidence?: number;
  };
}

/**
 * Enhanced real-time analysis hook that combines WebSocket updates
 * with polling fallback for maximum reliability
 */
export function useRealTimeAnalysis(options: RealTimeAnalysisOptions = {}) {
  const {
    jobId,
    onProgress,
    onComplete,
    onError,
    autoRetry = true,
    maxRetries = 3
  } = options;

  const { progress: wsProgress, isConnected } = useJobProgress(jobId || null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>({
    status: 'idle',
    progress: 0,
    stage: 'Initializing',
    message: 'Ready to analyze'
  });

  const [retryCount, setRetryCount] = useState(0);
  const [isPolling, setIsPolling] = useState(false);
  const pollingIntervalRef = useRef<TimeoutId | null>(null);
  const { toast } = useToast();

  // Update analysis state when WebSocket progress changes
  const normalizedJobId = jobId || null;
  
  useEffect(() => {
    if (wsProgress && wsProgress.id === normalizedJobId) {
      const newState: AnalysisState = {
        status: wsProgress.status === 'processing' ? 'analyzing' : 
               wsProgress.status === 'queued' ? 'connecting' :
               wsProgress.status as AnalysisState['status'],
        progress: wsProgress.percentage,
        stage: wsProgress.stage,
        message: wsProgress.message,
        startTime: wsProgress.startTime,
        endTime: wsProgress.endTime,
        estimatedTimeRemaining: wsProgress.estimatedTimeRemaining,
        error: wsProgress.error,
        result: wsProgress.result,
        metrics: wsProgress.metrics
      };

      setAnalysisState(newState);

      // Trigger callbacks
      if (wsProgress.status === 'completed' && onComplete) {
        onComplete(wsProgress.result);
      } else if (wsProgress.status === 'failed' && onError) {
        onError(wsProgress.error || 'Analysis failed');
      } else if (onProgress) {
        onProgress(wsProgress);
      }
    }
  }, [wsProgress, normalizedJobId, onProgress, onComplete, onError]);

  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearTimeout(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  // Fallback polling when WebSocket is not available
  const startPolling = useCallback(async () => {
    if (!jobId || isPolling) return;

    setIsPolling(true);
    logger.info('Starting fallback polling for job', { jobId });

    const pollJob = async () => {
      try {
        const response = await fetch(`/api/v2/analysis/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const jobData = await response.json();
        
        if (jobData.status === 'completed' || jobData.status === 'failed') {
          stopPolling();
          setAnalysisState(prev => ({
            ...prev,
            status: jobData.status,
            progress: 100,
            endTime: jobData.endTime || new Date().toISOString(),
            result: jobData.result,
            error: jobData.error
          }));

          if (jobData.status === 'completed' && onComplete) {
            onComplete(jobData.result);
          } else if (jobData.status === 'failed' && onError) {
            onError(jobData.error || 'Analysis failed');
          }
        } else {
          setAnalysisState(prev => ({
            ...prev,
            status: jobData.status === 'processing' ? 'analyzing' : 
                   jobData.status === 'queued' ? 'connecting' :
                   jobData.status as AnalysisState['status'],
            progress: jobData.progress || prev.progress,
            stage: jobData.stage || prev.stage,
            message: jobData.message || prev.message,
            estimatedTimeRemaining: jobData.estimatedTimeRemaining
          }));

          if (onProgress) {
            onProgress(jobData);
          }
        }
      } catch (error) {
        logger.error('Polling error', error instanceof Error ? error : new Error(String(error)));
        if (retryCount < maxRetries && autoRetry) {
          setRetryCount(prev => prev + 1);
        } else {
          stopPolling();
          setAnalysisState(prev => ({
            ...prev,
            status: 'failed',
            error: 'Failed to fetch analysis status'
          }));
          if (onError) {
            onError('Failed to fetch analysis status');
          }
        }
      }
    };

    // Initial poll
    await pollJob();

    // Set up interval polling - DISABLED to prevent spam
    // const scheduleNextPoll = () => {
    //   if (pollingIntervalRef.current) {
    //     pollingIntervalRef.current = setTimeout(() => {
    //       pollJob().then(() => {
    //         if (isPolling) {
    //           scheduleNextPoll();
    //         }
    //       });
    //     }, 5000); // Increased from 2000ms to 5000ms
    //   }
    // };
    // 
    // scheduleNextPoll();
  }, [jobId, isPolling, retryCount, maxRetries, autoRetry, onProgress, onComplete, onError, stopPolling]);

  // Start polling when WebSocket is not connected and we have a jobId
  useEffect(() => {
    if (jobId && !isConnected && !isPolling) {
      startPolling();
    } else if (isConnected && isPolling) {
      stopPolling();
    }

    return () => {
      stopPolling();
    };
  }, [jobId, isConnected, isPolling, startPolling, stopPolling]);

  // Cancel analysis
  const cancelAnalysis = useCallback(async () => {
    if (!jobId) return false;

    try {
      const response = await fetch(`/api/v2/analysis/jobs/${jobId}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to cancel analysis: ${response.statusText}`);
      }

      setAnalysisState(prev => ({
        ...prev,
        status: 'cancelled',
        message: 'Analysis cancelled by user'
      }));

      stopPolling();
      
      toast({
        title: 'Analysis Cancelled',
        description: 'The analysis has been cancelled successfully.',
        duration: 3000
      });

      return true;
    } catch (error) {
      logger.error('Failed to cancel analysis', error instanceof Error ? error : new Error(String(error)));
      toast({
        title: 'Cancellation Failed',
        description: 'Failed to cancel the analysis. Please try again.',
        variant: 'destructive',
        duration: 5000
      });
      return false;
    }
  }, [jobId, stopPolling, toast]);

  // Reset state
  const resetState = useCallback(() => {
    setAnalysisState({
      status: 'idle',
      progress: 0,
      stage: 'Initializing',
      message: 'Ready to analyze'
    });
    setRetryCount(0);
    stopPolling();
  }, [stopPolling]);

  // Get connection status
  const getConnectionStatus = useCallback(() => {
    if (isConnected) return 'websocket';
    if (isPolling) return 'polling';
    return 'disconnected';
  }, [isConnected, isPolling]);

  return {
    // State
    analysisState,
    isConnected,
    isPolling,
    connectionStatus: getConnectionStatus(),
    retryCount,

    // Actions
    cancelAnalysis,
    resetState,
    
    // Computed properties
    isAnalyzing: analysisState.status === 'analyzing',
    isCompleted: analysisState.status === 'completed',
    isFailed: analysisState.status === 'failed',
    isCancelled: analysisState.status === 'cancelled',
    isIdle: analysisState.status === 'idle',
    
    // Progress info
    progress: analysisState.progress,
    stage: analysisState.stage,
    message: analysisState.message,
    estimatedTimeRemaining: analysisState.estimatedTimeRemaining,
    
    // Results
    result: analysisState.result,
    error: analysisState.error,
    metrics: analysisState.metrics,
    
    // Timing
    startTime: analysisState.startTime,
    endTime: analysisState.endTime
  };
}

export default useRealTimeAnalysis;
