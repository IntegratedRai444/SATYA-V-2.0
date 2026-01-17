import React, { useState, useCallback } from 'react';

interface JobProgress {
  id: string;
  status: string;
  stage: string;
  percentage: number;
  message: string;
  startTime: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
  result?: any;
  error?: string;
}

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const [jobUpdates, setJobUpdates] = useState<Map<string, JobProgress>>(new Map());
  const [isConnected, setIsConnected] = useState(false);

  // Get WebSocket URL with auth token
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    return `${protocol}//${host}/ws`;
  }, []);

  // Handle job-related messages
  const handleJobMessage = useCallback((message: WebSocketMessage) => {
    const jobMessageTypes = [
      'job_started',
      'job_progress',
      'job_completed',
      'job_failed',
      'job_cancelled',
      'job_status'
    ];

    if (jobMessageTypes.includes(message.type) && message.data) {
      setJobUpdates(prev => {
        const newMap = new Map(prev);
        newMap.set(message.data.jobId, message.data);
        return newMap;
      });
    }
  }, []);

  // Subscribe to a specific job
  const subscribeToJob = useCallback((jobId: string) => {
    // Implementation would go here
    console.log('Subscribing to job:', jobId);
  }, []);

  // Unsubscribe from a specific job
  const unsubscribeFromJob = useCallback((jobId: string) => {
    // Implementation would go here
    console.log('Unsubscribing from job:', jobId);
  }, []);

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
    isConnected,
    jobUpdates,
    subscribeToJob,
    unsubscribeFromJob,
    getJobProgress,
    clearJobProgress,
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