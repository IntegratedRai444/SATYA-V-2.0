import { useEffect, useRef, useState, useCallback } from 'react';
import apiClient from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';

export interface DashboardUpdateMessage {
  type: 'stats' | 'analytics' | 'activity' | 'metrics' | 'all';
  data: any;
  timestamp: string;
}

interface UseDashboardWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  onStatsUpdate?: (data: any) => void;
  onAnalyticsUpdate?: (data: any) => void;
  onActivityUpdate?: (data: any) => void;
  onMetricsUpdate?: (data: any) => void;
}

export function useDashboardWebSocket(options: UseDashboardWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 5000,
    onStatsUpdate,
    onAnalyticsUpdate,
    onActivityUpdate,
    onMetricsUpdate
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const queryClient = useQueryClient();

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const token = apiClient.getAuthToken();
    return `${protocol}//${host}/api/v1/dashboard/ws?token=${token}`;
  }, []);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: DashboardUpdateMessage = JSON.parse(event.data);
      
      // Invalidate relevant queries to trigger refetch
      switch (message.type) {
        case 'stats':
          queryClient.invalidateQueries({ queryKey: ['dashboard', 'stats'] });
          onStatsUpdate?.(message.data);
          break;
        case 'analytics':
          queryClient.invalidateQueries({ queryKey: ['dashboard', 'analytics'] });
          onAnalyticsUpdate?.(message.data);
          break;
        case 'activity':
          queryClient.invalidateQueries({ queryKey: ['dashboard', 'activity'] });
          onActivityUpdate?.(message.data);
          break;
        case 'metrics':
          queryClient.invalidateQueries({ queryKey: ['dashboard', 'metrics'] });
          onMetricsUpdate?.(message.data);
          break;
        case 'all':
          queryClient.invalidateQueries({ queryKey: ['dashboard'] });
          onStatsUpdate?.(message.data.stats);
          onAnalyticsUpdate?.(message.data.analytics);
          onActivityUpdate?.(message.data.activity);
          onMetricsUpdate?.(message.data.metrics);
          break;
      }
    } catch (err) {
      console.error('Error processing WebSocket message:', err);
    }
  }, [onActivityUpdate, onAnalyticsUpdate, onMetricsUpdate, onStatsUpdate, queryClient]);

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const wsUrl = getWebSocketUrl();
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      reconnectCountRef.current = 0;
      
      // Subscribe to all dashboard updates
      ws.send(JSON.stringify({ type: 'subscribe', channels: ['stats', 'analytics', 'activity', 'metrics'] }));
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectCountRef.current < reconnectAttempts) {
        reconnectCountRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
      } else {
        setError('Failed to connect to dashboard updates. Please refresh the page to try again.');
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Attempting to reconnect...');
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [getWebSocketUrl, handleMessage, reconnectAttempts, reconnectInterval]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [autoConnect, connect]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  const subscribeToChannel = useCallback((channel: string) => {
    return sendMessage({ type: 'subscribe', channel });
  }, [sendMessage]);

  const unsubscribeFromChannel = useCallback((channel: string) => {
    return sendMessage({ type: 'unsubscribe', channel });
  }, [sendMessage]);

  return {
    isConnected,
    error,
    connect,
    disconnect: () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    },
    sendMessage,
    subscribeToChannel,
    unsubscribeFromChannel,
    reconnectCount: reconnectCountRef.current,
  };
}
