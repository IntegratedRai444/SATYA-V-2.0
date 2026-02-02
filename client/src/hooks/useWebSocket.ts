import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';
import logger from '../lib/logger';

export interface JobProgress {
  id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  stage: string;
  percentage: number;
  message: string;
  startTime: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
  result?: unknown;
  error?: string;
  metrics?: {
    processingTime?: number;
    modelVersion?: string;
    confidence?: number;
  };
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  enableMessageRouting?: boolean;
}

// Type guard functions
function isJobProgressMessage(message: WebSocketMessage): message is WebSocketMessage & {
  payload: { jobId: string; status?: string; stage?: string; progress?: number; message?: string; startTime?: string; endTime?: string; estimatedTimeRemaining?: number; result?: unknown; error?: string; }
} {
  return ['JOB_PROGRESS', 'JOB_COMPLETED', 'JOB_FAILED', 'JOB_STARTED', 'JOB_STAGE_UPDATE', 'JOB_METRICS'].includes(message.type) &&
         typeof message.payload === 'object' &&
         message.payload !== null &&
         'jobId' in message.payload &&
         typeof (message.payload as { jobId: unknown }).jobId === 'string';
}

function isDashboardUpdateMessage(message: WebSocketMessage): boolean {
  return message.type === 'DASHBOARD_UPDATE';
}

function isNotificationMessage(message: WebSocketMessage): boolean {
  return message.type === 'notification';
}

function isSystemAlertMessage(message: WebSocketMessage): boolean {
  return message.type === 'system_alert';
}

// Message router singleton
class WebSocketMessageRouter {
  private static instance: WebSocketMessageRouter;
  private handlers = new Map<string, Set<(message: WebSocketMessage) => void>>();
  
  static getInstance(): WebSocketMessageRouter {
    if (!WebSocketMessageRouter.instance) {
      WebSocketMessageRouter.instance = new WebSocketMessageRouter();
    }
    return WebSocketMessageRouter.instance;
  }
  
  register(type: string, handler: (message: WebSocketMessage) => void): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
    
    return () => {
      this.handlers.get(type)?.delete(handler);
    };
  }
  
  route(message: WebSocketMessage): void {
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          logger.error('Error in message handler', error as Error);
        }
      });
    }
  }
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { enableMessageRouting = true, autoConnect, reconnectAttempts, reconnectInterval } = options;
  const [jobUpdates, setJobUpdates] = useState<Map<string, JobProgress>>(new Map());
  const [router] = useState(() => WebSocketMessageRouter.getInstance());
  const cleanupRef = useRef<(() => void) | null>(null);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      // Route message if routing is enabled
      if (enableMessageRouting) {
        router.route(message);
      }
      
      // Handle job-related messages
      if (isJobProgressMessage(message)) {
        const payload = message.payload;
        const jobId = payload.jobId;
        
        const progressData: JobProgress = {
          id: jobId,
          status: message.type === 'JOB_COMPLETED' ? 'completed' : 
                 message.type === 'JOB_FAILED' ? 'failed' : 
                 message.type === 'JOB_STARTED' ? 'processing' :
                 (payload.status || 'processing') as JobProgress['status'],
          stage: payload.stage || 'processing',
          percentage: payload.progress || 0,
          message: payload.message || '',
          startTime: payload.startTime || new Date().toISOString(),
          endTime: payload.endTime,
          estimatedTimeRemaining: payload.estimatedTimeRemaining,
          result: payload.result,
          error: payload.error
        };

        setJobUpdates(prev => {
          const newMap = new Map(prev);
          newMap.set(jobId, progressData);
          return newMap;
        });
      }
      
      // Log other message types for debugging
      if (isDashboardUpdateMessage(message)) {
        logger.debug('Dashboard update received', message);
      } else if (isNotificationMessage(message)) {
        logger.debug('Notification received', message);
      } else if (isSystemAlertMessage(message)) {
        logger.debug('System alert received', message);
      }
    } catch (error) {
      logger.error('Error handling WebSocket message', error as Error);
    }
  }, [enableMessageRouting, router]);

  // Use base WebSocket with authentication
  const base = useBaseWebSocket({
    autoConnect,
    reconnectAttempts,
    reconnectInterval,
    onMessage: handleMessage,
  });

  // Subscribe to a specific job
  const subscribeToJob = useCallback((jobId: string) => {
    if (!base.isConnected) {
      logger.warn('WebSocket not connected, cannot subscribe to job', { jobId });
      return false;
    }
    
    return base.sendMessage({
      type: 'SUBSCRIBE_JOB',
      payload: { jobId },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [base]);

  // Unsubscribe from a specific job
  const unsubscribeFromJob = useCallback((jobId: string) => {
    if (!base.isConnected) {
      logger.warn('WebSocket not connected, cannot unsubscribe from job', { jobId });
      return false;
    }
    
    return base.sendMessage({
      type: 'UNSUBSCRIBE_JOB',
      payload: { jobId },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
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

  // Register message handler for external hooks
  const registerMessageHandler = useCallback((type: string, handler: (message: WebSocketMessage) => void) => {
    return router.register(type, handler);
  }, [router]);

  // Cleanup on unmount
  useEffect(() => {
    const cleanup = cleanupRef.current;
    return () => {
      if (cleanup) {
        cleanup();
      }
    };
  }, []);

  // Memoize router value to avoid ref access during render
  const routerValue = useMemo(() => router, [router]);

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
    registerMessageHandler,
    router: routerValue
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