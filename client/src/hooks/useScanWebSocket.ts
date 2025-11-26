import { useCallback, useMemo } from 'react';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';
import { useToast } from '@/components/ui/use-toast';

export interface ScanUpdateMessage extends WebSocketMessage {
  scanId: string;
  status: string;
  progress?: number;
  result?: any;
}

interface UseScanWebSocketProps {
  onScanUpdate?: (message: ScanUpdateMessage) => void;
  onError?: (error: Error) => void;
  autoConnect?: boolean;
  maxRetryAttempts?: number;
  retryDelay?: number;
}

/**
 * Custom hook for managing WebSocket connections for scan updates
 * Built on top of useBaseWebSocket with scan-specific functionality
 */
export function useScanWebSocket({
  onScanUpdate,
  onError,
  autoConnect = true,
  maxRetryAttempts = 3,
  retryDelay = 3000,
}: UseScanWebSocketProps = {}) {
  const { toast } = useToast();

  // Handle scan-specific messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'scan_update' && onScanUpdate) {
      onScanUpdate(message as ScanUpdateMessage);
    }
  }, [onScanUpdate]);

  // Handle errors with toast notifications
  const handleError = useCallback((error: Error) => {
    if (onError) {
      onError(error);
    } else {
      toast({
        title: 'Connection Error',
        description: error.message || 'An error occurred with the real-time connection',
        variant: 'destructive',
        duration: 5000,
      });
    }
  }, [onError, toast]);

  // Use base WebSocket with scan-specific configuration
  const base = useBaseWebSocket({
    autoConnect,
    reconnectAttempts: maxRetryAttempts,
    reconnectInterval: retryDelay,
    onMessage: handleMessage,
    onError: handleError,
  });

  // Subscribe to a scan's updates
  const subscribeToScan = useCallback((scanId: string) => {
    if (!scanId) {
      console.warn('Attempted to subscribe to scan with empty ID');
      return () => { };
    }

    base.sendMessage({
      type: 'subscribe_scan',
      scanId
    });

    // Return unsubscribe function
    return () => {
      base.sendMessage({
        type: 'unsubscribe_scan',
        scanId
      });
    };
  }, [base]);

  // Unsubscribe from a scan's updates
  const unsubscribeFromScan = useCallback((scanId: string) => {
    if (!scanId) return;

    base.sendMessage({
      type: 'unsubscribe_scan',
      scanId
    });
  }, [base]);

  // Memoize the API to prevent unnecessary re-renders
  const api = useMemo(() => ({
    subscribeToScan,
    unsubscribeFromScan,
    isConnected: base.isConnected,
    connectionStatus: base.connectionStatus,
    sendMessage: base.sendMessage,
    reconnect: base.reconnect,
  }), [subscribeToScan, unsubscribeFromScan, base]);

  return api;
}
