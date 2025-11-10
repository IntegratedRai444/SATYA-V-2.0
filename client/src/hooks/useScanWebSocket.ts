import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { webSocketService, WebSocketMessage, ScanUpdateMessage } from '@/services/websocket';
import { useToast } from '@/components/ui/use-toast';

type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';

interface UseScanWebSocketProps {
  /** Callback when a scan update is received */
  onScanUpdate?: (message: ScanUpdateMessage) => void;
  /** Callback when an error occurs */
  onError?: (error: Error) => void;
  /** Whether to automatically connect to the WebSocket */
  autoConnect?: boolean;
  /** Maximum number of connection retry attempts */
  maxRetryAttempts?: number;
  /** Delay between retry attempts in milliseconds */
  retryDelay?: number;
}

/**
 * Custom hook for managing WebSocket connections for scan updates
 */
export function useScanWebSocket({
  onScanUpdate,
  onError,
  autoConnect = true,
  maxRetryAttempts = 3,
  retryDelay = 3000,
}: UseScanWebSocketProps = {}) {
  const { toast } = useToast();
  const subscriptions = useRef<Record<string, () => void>>({});
  const isMounted = useRef(true);
  const retryCount = useRef(0);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const retryTimeoutRef = useRef<NodeJS.Timeout>();

  // Update connection status when it changes
  const updateConnectionStatus = useCallback((status: ConnectionStatus) => {
    if (isMounted.current) {
      setConnectionStatus(status);
    }
  }, []);

  // Handle WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      if (message.type === 'scan_update' && onScanUpdate) {
        onScanUpdate(message as ScanUpdateMessage);
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      handleError(error instanceof Error ? error : new Error('Error processing WebSocket message'));
    }
  }, [onScanUpdate]);

  // Handle WebSocket errors
  const handleError = useCallback((error: Error) => {
    console.error('WebSocket error:', error);
    updateConnectionStatus('error');
    
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
  }, [onError, toast, updateConnectionStatus]);

  // Subscribe to a scan's updates
  const subscribeToScan = useCallback((scanId: string) => {
    if (!scanId) {
      console.warn('Attempted to subscribe to scan with empty ID');
      return () => {};
    }
    
    if (subscriptions.current[scanId]) {
      console.log(`Already subscribed to scan ${scanId}`);
      return () => unsubscribeFromScan(scanId);
    }
    
    try {
      // Subscribe to WebSocket channel
      webSocketService.subscribeToScan(scanId);
      
      // Track the subscription for cleanup
      const unsubscribe = webSocketService.subscribe((message: WebSocketMessage) => {
        if (message.type === 'scan_update' && 'scanId' in message && message.scanId === scanId) {
          handleMessage(message);
        }
      });
      
      // Store the unsubscribe function
      subscriptions.current[scanId] = () => {
        try {
          unsubscribe();
          webSocketService.unsubscribeFromScan(scanId);
        } catch (error) {
          console.error(`Error unsubscribing from scan ${scanId}:`, error);
        } finally {
          delete subscriptions.current[scanId];
        }
      };
      
      console.log(`Subscribed to scan ${scanId}`);
      return subscriptions.current[scanId];
    } catch (error) {
      console.error(`Failed to subscribe to scan ${scanId}:`, error);
      handleError(error instanceof Error ? error : new Error(`Failed to subscribe to scan ${scanId}`));
      return () => {};
    }
  }, [handleMessage, handleError]);

  // Unsubscribe from a scan's updates
  const unsubscribeFromScan = useCallback((scanId: string) => {
    if (!scanId) return;
    
    const unsubscribe = subscriptions.current[scanId];
    if (unsubscribe) {
      try {
        unsubscribe();
      } catch (error) {
        console.error(`Error unsubscribing from scan ${scanId}:`, error);
      } finally {
        delete subscriptions.current[scanId];
      }
    }
  }, []);

  // Handle connection status changes
  useEffect(() => {
    const handleConnected = () => {
      updateConnectionStatus('connected');
      retryCount.current = 0; // Reset retry count on successful connection
    };
    
    const handleDisconnected = () => {
      updateConnectionStatus('disconnected');
    };
    
    const handleConnecting = () => {
      updateConnectionStatus('connecting');
    };
    
    // Set up event listeners
    const connectedUnsubscribe = webSocketService.on('connected', handleConnected);
    const disconnectedUnsubscribe = webSocketService.on('disconnected', handleDisconnected);
    const reconnectingUnsubscribe = webSocketService.on('reconnecting', handleConnecting);
    
    return () => {
      connectedUnsubscribe();
      disconnectedUnsubscribe();
      reconnectingUnsubscribe();
    };
  }, [updateConnectionStatus]);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      isMounted.current = false;
      
      // Clear any pending retry timeouts
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      
      // Clean up all subscriptions
      Object.values(subscriptions.current).forEach(unsubscribe => {
        try {
          unsubscribe();
        } catch (error) {
          console.error('Error during unsubscribe cleanup:', error);
        }
      });
      
      // Clear the subscriptions
      subscriptions.current = {};
    };
  }, []);

  // Auto-connect if enabled
  useEffect(() => {
    if (!autoConnect) return;
    
    const connectWithRetry = async (attempt = 0) => {
      if (attempt >= maxRetryAttempts) {
        handleError(new Error(`Failed to connect after ${maxRetryAttempts} attempts`));
        return;
      }
      
      try {
        updateConnectionStatus(attempt === 0 ? 'connecting' : 'reconnecting');
        await webSocketService.connect();
        retryCount.current = 0;
      } catch (error) {
        console.error(`Connection attempt ${attempt + 1} failed:`, error);
        retryCount.current = attempt + 1;
        
        // Schedule a retry with exponential backoff
        const delay = Math.min(retryDelay * Math.pow(2, attempt), 30000); // Cap at 30 seconds
        
        if (isMounted.current) {
          retryTimeoutRef.current = setTimeout(() => {
            if (isMounted.current) {
              connectWithRetry(attempt + 1);
            }
          }, delay);
        }
      }
    };
    
    connectWithRetry(0);
    
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      
      if (autoConnect && isMounted.current) {
        webSocketService.disconnect();
      }
    };
  }, [autoConnect, handleError, maxRetryAttempts, retryDelay, updateConnectionStatus]);

  // Memoize the API to prevent unnecessary re-renders
  const api = useMemo(() => ({
    subscribeToScan,
    unsubscribeFromScan,
    isConnected: connectionStatus === 'connected',
    connectionStatus,
    sendMessage: webSocketService.send.bind(webSocketService),
    reconnect: () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      webSocketService.connect().catch(handleError);
    },
  }), [subscribeToScan, unsubscribeFromScan, connectionStatus, handleError]);
  
  return api;
}
