import { useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';

export interface DashboardUpdateMessage extends WebSocketMessage {
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

  const queryClient = useQueryClient();

  // Handle dashboard-specific messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    const dashboardMessage = message as DashboardUpdateMessage;

    // Invalidate relevant queries to trigger refetch
    switch (dashboardMessage.type) {
      case 'stats':
        queryClient.invalidateQueries({ queryKey: ['dashboard', 'stats'] });
        onStatsUpdate?.(dashboardMessage.data);
        break;
      case 'analytics':
        queryClient.invalidateQueries({ queryKey: ['dashboard', 'analytics'] });
        onAnalyticsUpdate?.(dashboardMessage.data);
        break;
      case 'activity':
        queryClient.invalidateQueries({ queryKey: ['dashboard', 'activity'] });
        onActivityUpdate?.(dashboardMessage.data);
        break;
      case 'metrics':
        queryClient.invalidateQueries({ queryKey: ['dashboard', 'metrics'] });
        onMetricsUpdate?.(dashboardMessage.data);
        break;
      case 'all':
        queryClient.invalidateQueries({ queryKey: ['dashboard'] });
        onStatsUpdate?.(dashboardMessage.data.stats);
        onAnalyticsUpdate?.(dashboardMessage.data.analytics);
        onActivityUpdate?.(dashboardMessage.data.activity);
        onMetricsUpdate?.(dashboardMessage.data.metrics);
        break;
    }
  }, [onActivityUpdate, onAnalyticsUpdate, onMetricsUpdate, onStatsUpdate, queryClient]);

  // Use base WebSocket with dashboard-specific configuration
  const base = useBaseWebSocket({
    autoConnect,
    reconnectAttempts,
    reconnectInterval,
    onMessage: handleMessage,
  });

  // Subscribe to dashboard channels on connection
  const subscribeToChannel = useCallback((channel: string) => {
    return base.sendMessage({ type: 'subscribe', channel });
  }, [base]);

  // Unsubscribe from dashboard channels
  const unsubscribeFromChannel = useCallback((channel: string) => {
    return base.sendMessage({ type: 'unsubscribe', channel });
  }, [base]);

  return {
    isConnected: base.isConnected,
    connectionError: base.connectionError,
    connectionStatus: base.connectionStatus,
    connect: base.connect,
    disconnect: base.disconnect,
    sendMessage: base.sendMessage,
    subscribeToChannel,
    unsubscribeFromChannel,
    reconnect: base.reconnect,
  };
}
