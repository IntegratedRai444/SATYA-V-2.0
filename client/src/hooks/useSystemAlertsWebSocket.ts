import { useCallback, useEffect } from 'react';
import { WebSocketMessage } from './useBaseWebSocket';
import { useWebSocket } from './useWebSocket';
import logger from '../lib/logger';

export interface SystemAlertMessage extends WebSocketMessage {
  type: 'system_alert';
  payload: {
    id: string;
    title: string;
    message: string;
    severity: 'info' | 'warning' | 'error' | 'critical';
    timestamp: string;
    action?: {
      label: string;
      url?: string;
      callback?: string;
    };
    autoClose?: boolean;
    duration?: number;
    category?: 'system' | 'security' | 'performance' | 'maintenance';
  };
}

interface UseSystemAlertsWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  onSystemAlert?: (alert: SystemAlertMessage['payload']) => void;
  onAlertAcknowledged?: (alertId: string) => void;
}

export function useSystemAlertsWebSocket(options: UseSystemAlertsWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 5000,
    onSystemAlert,
    onAlertAcknowledged
  } = options;

  const { registerMessageHandler } = useWebSocket({ autoConnect, reconnectAttempts, reconnectInterval });

  // Handle system alert messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      if (message.type === 'system_alert') {
        const alertMessage = message as SystemAlertMessage;
        logger.debug('System alert received', { 
          id: alertMessage.payload.id,
          severity: alertMessage.payload.severity,
          title: alertMessage.payload.title
        });
        
        onSystemAlert?.(alertMessage.payload);
      } else if (message.type === 'alert_acknowledged') {
        const alertId = (message.payload as { alertId?: string })?.alertId;
        if (alertId) {
          logger.debug('System alert acknowledged', { alertId });
          onAlertAcknowledged?.(alertId);
        }
      }
    } catch (error) {
      logger.error('Error handling system alert message', error instanceof Error ? error : new Error(String(error)));
    }
  }, [onSystemAlert, onAlertAcknowledged]);

  // Register message handler for system alerts
  useEffect(() => {
    const cleanup = registerMessageHandler('system_alert', handleMessage);
    return cleanup;
  }, [registerMessageHandler, handleMessage]);

  // Get connection status from useWebSocket
  const { isConnected, connectionError, connectionStatus, connect, disconnect, sendMessage } = useWebSocket({ autoConnect, reconnectAttempts, reconnectInterval });

  // Reconnect function
  const reconnect = useCallback(() => {
    return connect();
  }, [connect]);

  // Subscribe to system alerts channel on connection
  const subscribeToAlerts = useCallback(() => {
    if (!isConnected) {
      logger.warn('WebSocket not connected, cannot subscribe to system alerts');
      return false;
    }
    
    return sendMessage({
      type: 'subscribe',
      payload: { channel: 'system_alerts' },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [isConnected, sendMessage]);

  // Acknowledge alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    if (!isConnected) {
      logger.warn('WebSocket not connected, cannot acknowledge alert');
      return false;
    }
    
    return sendMessage({
      type: 'alert_acknowledged',
      payload: { alertId },
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
    subscribeToAlerts,
    acknowledgeAlert,
    reconnect,
  };
}
