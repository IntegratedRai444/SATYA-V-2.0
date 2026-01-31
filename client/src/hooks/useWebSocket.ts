import React, { useState, useCallback } from 'react';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';

interface JobProgress {
  id: string;
  status: string;
  stage: string;
  percentage: number;
  message: string;
  startTime: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
  result?: unknown;
  error?: string;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const [jobUpdates, setJobUpdates] = useState<Map<string, JobProgress>>(new Map());

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'JOB_PROGRESS' || message.type === 'JOB_COMPLETED' || message.type === 'JOB_FAILED') {
      const payload = message.payload as Record<string, unknown>;
      const jobId = payload?.jobId as string;
      if (jobId) {
        const progressData = {
          id: jobId,
          status: message.type === 'JOB_COMPLETED' ? 'completed' : 
                 message.type === 'JOB_FAILED' ? 'failed' : 
                 (payload?.status as string) || 'processing',
          stage: (payload?.stage as string) || 'processing',
          percentage: (payload?.progress as number) || 0,
          message: (payload?.message as string) || '',
          startTime: (payload?.startTime as string) || new Date().toISOString(),
          endTime: payload?.endTime as string,
          estimatedTimeRemaining: payload?.estimatedTimeRemaining as number,
          result: payload?.result,
          error: payload?.error as string
        };

        setJobUpdates(prev => {
          const newMap = new Map(prev);
          newMap.set(jobId, progressData);
          return newMap;
        });
      }
    }
  }, []);

  // Use base WebSocket with authentication
  const base = useBaseWebSocket({
    autoConnect: options.autoConnect,
    reconnectAttempts: options.reconnectAttempts,
    reconnectInterval: options.reconnectInterval,
    onMessage: handleMessage,
  });

  // Subscribe to a specific job
  const subscribeToJob = useCallback((jobId: string) => {
    if (!base.isConnected) {
      console.warn('WebSocket not connected, cannot subscribe to job');
      return;
    }
    
    base.sendMessage({
      type: 'SUBSCRIBE_JOB',
      jobId,
      timestamp: new Date().toISOString()
    });
  }, [base]);

  // Unsubscribe from a specific job
  const unsubscribeFromJob = useCallback((jobId: string) => {
    if (!base.isConnected) {
      console.warn('WebSocket not connected, cannot unsubscribe from job');
      return;
    }
    
    base.sendMessage({
      type: 'UNSUBSCRIBE_JOB',
      jobId,
      timestamp: new Date().toISOString()
    });
  }, [base]);

  // Get progress for a specific job
  const getJobProgress = useCallback((jobId: string) => {
    return jobUpdates.get(jobId);
  }, [jobUpdates]);

  // Clear progress for a specific job
  const clearJobProgress = useCallback((jobId: string) => {
    setJobUpdates(prev => {
      const newMap = new Map(prev);
      newMap.delete(jobId);
      return newMap;
    });
  }, []);

  return {
    isConnected: base.isConnected,
    jobUpdates,
    subscribeToJob,
    unsubscribeFromJob,
    getJobProgress,
    clearJobProgress,
    sendMessage: base.sendMessage,
    connectionError: base.connectionError,
    connectionStatus: base.connectionStatus,
    connect: base.connect,
    disconnect: base.disconnect,
  };
}

// Hook for tracking a specific job
export function useJobProgress(jobId: string | null) {
  const { isConnected, subscribeToJob, unsubscribeFromJob, getJobProgress } = useWebSocket();
  const [progress, setProgress] = useState<JobProgress | null>(null);

  // Subscribe/unsubscribe when jobId changes
  React.useEffect(() => {
    if (jobId && isConnected) {
      subscribeToJob(jobId);

      return () => {
        unsubscribeFromJob(jobId);
      };
    }
  }, [jobId, isConnected, subscribeToJob, unsubscribeFromJob]);

  // Update progress when it changes
  React.useEffect(() => {
    if (jobId) {
      const jobProgress = getJobProgress(jobId);
      setProgress(jobProgress || null);
    }
  }, [jobId, getJobProgress]);

  return {
    progress,
    isConnected,
    isCompleted: progress?.status === 'completed',
    isFailed: progress?.status === 'failed',
    isProcessing: progress?.status === 'processing',
    isQueued: progress?.status === 'queued',
    isCancelled: progress?.status === 'cancelled'
  };
}