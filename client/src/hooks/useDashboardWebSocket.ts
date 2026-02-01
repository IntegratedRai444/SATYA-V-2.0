import { useCallback, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { WebSocketMessage } from './useBaseWebSocket';
import { useWebSocket } from './useWebSocket';
import logger from '../lib/logger';

export interface DashboardUpdateMessage extends WebSocketMessage {
  type: 'DASHBOARD_UPDATE';
  payload: {
    updateType: 'stats' | 'analytics' | 'activity' | 'metrics' | 'all';
    data: unknown;
    timestamp: string;
  };
}

interface UseDashboardWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  onStatsUpdate?: (data: unknown) => void;
  onAnalyticsUpdate?: (data: unknown) => void;
  onActivityUpdate?: (data: unknown) => void;
  onMetricsUpdate?: (data: unknown) => void;
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

  const queryClient = useQueryClient();
  const { registerMessageHandler } = useWebSocket({ autoConnect, reconnectAttempts, reconnectInterval });

  // Handle dashboard-specific messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      if (message.type === 'DASHBOARD_UPDATE') {
        const dashboardMessage = message as DashboardUpdateMessage;
        const { updateType, data } = dashboardMessage.payload;

        logger.debug('Dashboard update received', { updateType, hasData: !!data });

        // Invalidate relevant queries to trigger refetch
        switch (updateType) {
          case 'stats':
            queryClient.invalidateQueries({ queryKey: ['dashboard', 'stats'] });
            onStatsUpdate?.(data);
            break;
          case 'analytics':
            queryClient.invalidateQueries({ queryKey: ['dashboard', 'analytics'] });
            onAnalyticsUpdate?.(data);
            break;
          case 'activity':
            queryClient.invalidateQueries({ queryKey: ['dashboard', 'activity'] });
            onActivityUpdate?.(data);
            break;
          case 'metrics':
            queryClient.invalidateQueries({ queryKey: ['dashboard', 'metrics'] });
            onMetricsUpdate?.(data);
            break;
          case 'all':
            queryClient.invalidateQueries({ queryKey: ['dashboard'] });
            onStatsUpdate?.((data as { stats?: unknown })?.stats);
            onAnalyticsUpdate?.((data as { analytics?: unknown })?.analytics);
            onActivityUpdate?.((data as { activity?: unknown })?.activity);
            onMetricsUpdate?.((data as { metrics?: unknown })?.metrics);
            break;
        }
      }
    } catch (error) {
      logger.error('Error handling dashboard message', error instanceof Error ? error : new Error(String(error)));
    }
  }, [onActivityUpdate, onAnalyticsUpdate, onMetricsUpdate, onStatsUpdate, queryClient]);

  // Register message handler for dashboard updates
  useEffect(() => {
    const cleanup = registerMessageHandler('DASHBOARD_UPDATE', handleMessage);
    return cleanup;
  }, [registerMessageHandler, handleMessage]);

  // Get connection status from useWebSocket
  const { isConnected, connectionError, connectionStatus, connect, disconnect, sendMessage } = useWebSocket({ autoConnect, reconnectAttempts, reconnectInterval });

  // Reconnect function
  const reconnect = useCallback(() => {
    return connect();
  }, [connect]);

  // Subscribe to dashboard channels on connection
  const subscribeToChannel = useCallback((channel: string) => {
    if (!isConnected) {
      logger.warn('WebSocket not connected, cannot subscribe to channel');
      return false;
    }
    
    return sendMessage({ 
      type: 'subscribe', 
      payload: { channel },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [isConnected, sendMessage]);

  // Unsubscribe from dashboard channels
  const unsubscribeFromChannel = useCallback((channel: string) => {
    if (!isConnected) {
      logger.warn('WebSocket not connected, cannot unsubscribe from channel');
      return false;
    }
    
    return sendMessage({ 
      type: 'unsubscribe', 
      payload: { channel },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [isConnected, sendMessage]);

  return {
    isConnected,
    connectionError,
    connectionStatus,
    connect,
    disconnect,
    sendMessage,
    subscribeToChannel,
    unsubscribeFromChannel,
    reconnect,
  };
}
