import { useEffect, useRef, useState, useCallback } from 'react';
import apiClient from '../lib/api';

export interface WebSocketMessage {
  type: string;
  data?: any;
  error?: string;
}

export interface JobProgress {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: {
    stage: string;
    percentage: number;
    message: string;
    startTime: string;
    endTime?: string;
    estimatedTimeRemaining?: number;
  };
  result?: any;
  error?: string;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [jobUpdates, setJobUpdates] = useState<Map<string, JobProgress>>(new Map());
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const subscribedJobsRef = useRef<Set<string>>(new Set());

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const token = apiClient.getAuthToken();
    
    if (!token) {
      throw new Error('No authentication token available');
    }
    
    return `${protocol}//${host}/ws?token=${encodeURIComponent(token)}`;
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = getWebSocketUrl();
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        reconnectCountRef.current = 0;
        
        // Resubscribe to jobs
        subscribedJobsRef.current.forEach(jobId => {
          ws.send(JSON.stringify({
            type: 'subscribe_job',
            jobId
          }));
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;
        
        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          scheduleReconnect();
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('Connection error occurred');
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionError((error as Error).message);
    }
  }, [getWebSocketUrl, reconnectAttempts]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    reconnectCountRef.current++;
    const delay = reconnectInterval * Math.pow(2, reconnectCountRef.current - 1); // Exponential backoff
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectCountRef.current}/${reconnectAttempts})`);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  }, [connect, reconnectInterval, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    reconnectCountRef.current = 0;
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  const subscribeToJob = useCallback((jobId: string) => {
    subscribedJobsRef.current.add(jobId);
    sendMessage({
      type: 'subscribe_job',
      jobId
    });
  }, [sendMessage]);

  const unsubscribeFromJob = useCallback((jobId: string) => {
    subscribedJobsRef.current.delete(jobId);
    sendMessage({
      type: 'unsubscribe_job',
      jobId
    });
  }, [sendMessage]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'connected':
        console.log('WebSocket connection confirmed:', message.data);
        break;

      case 'job_started':
      case 'job_progress':
      case 'job_completed':
      case 'job_failed':
      case 'job_cancelled':
        if (message.data) {
          setJobUpdates(prev => {
            const newMap = new Map(prev);
            newMap.set(message.data.jobId, message.data);
            return newMap;
          });
        }
        break;

      case 'job_status':
        // Initial job status when subscribing
        if (message.data) {
          setJobUpdates(prev => {
            const newMap = new Map(prev);
            newMap.set(message.data.jobId, message.data);
            return newMap;
          });
        }
        break;

      case 'error':
        console.error('WebSocket error message:', message.error);
        setConnectionError(message.error || 'Unknown error');
        break;

      case 'pong':
        // Heartbeat response
        break;

      default:
        console.log('Unknown WebSocket message type:', message.type);
    }
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && apiClient.isAuthenticated()) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    connectionError,
    jobUpdates,
    connect,
    disconnect,
    subscribeToJob,
    unsubscribeFromJob,
    sendMessage,
    getJobProgress: (jobId: string) => jobUpdates.get(jobId),
    clearJobProgress: (jobId: string) => {
      setJobUpdates(prev => {
        const newMap = new Map(prev);
        newMap.delete(jobId);
        return newMap;
      });
    }
  };
}

// Hook for tracking a specific job
export function useJobProgress(jobId: string | null) {
  const { isConnected, subscribeToJob, unsubscribeFromJob, getJobProgress } = useWebSocket();
  const [progress, setProgress] = useState<JobProgress | null>(null);

  useEffect(() => {
    if (jobId && isConnected) {
      subscribeToJob(jobId);
      
      return () => {
        unsubscribeFromJob(jobId);
      };
    }
  }, [jobId, isConnected, subscribeToJob, unsubscribeFromJob]);

  useEffect(() => {
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