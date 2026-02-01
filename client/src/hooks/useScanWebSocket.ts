import { useCallback, useEffect, useRef } from 'react';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';
import { useToast } from '@/components/ui/use-toast';
import logger from '../lib/logger';

export interface ScanUpdateMessage extends WebSocketMessage {
  scanId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress?: number;
  stage?: string;
  message?: string;
  startTime?: string;
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

interface UseScanWebSocketProps {
  onScanUpdate?: (message: ScanUpdateMessage) => void;
  onError?: (error: Error) => void;
  onScanCompleted?: (scanId: string, result: unknown) => void;
  onScanFailed?: (scanId: string, error: string) => void;
  autoConnect?: boolean;
  maxRetryAttempts?: number;
  retryDelay?: number;
  enableAutoRetry?: boolean;
}

/**
 * Custom hook for managing WebSocket connections for scan updates
 * Built on top of useBaseWebSocket with scan-specific functionality
 */

export function useScanWebSocket({
  onScanUpdate,
  onError,
  onScanCompleted,
  onScanFailed,
  autoConnect = true,
  maxRetryAttempts = 3,
  retryDelay = 3000,
  enableAutoRetry = true,
}: UseScanWebSocketProps = {}) {
  const { toast } = useToast();
  const subscribedScans = useRef<Set<string>>(new Set());
  const retryCountRef = useRef<Map<string, number>>(new Map());
  const baseRef = useRef<{ sendMessage: (message: unknown) => boolean } | null>(null);

  // Handle scan-specific messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'scan_update' || message.type === 'scan_completed' || message.type === 'scan_failed') {
      const scanMessage = message as ScanUpdateMessage;
      
      logger.debug('Scan update received', {
        scanId: scanMessage.scanId,
        status: scanMessage.status,
        progress: scanMessage.progress
      });

      // Trigger callbacks
      if (onScanUpdate) {
        onScanUpdate(scanMessage);
      }

      // Handle completion
      if (message.type === 'scan_completed' && onScanCompleted) {
        onScanCompleted(scanMessage.scanId, scanMessage.result);
      }

      // Handle failure
      if (message.type === 'scan_failed' && onScanFailed) {
        onScanFailed(scanMessage.scanId, scanMessage.error || 'Scan failed');
      }

      // Auto-retry failed scans
      if (message.type === 'scan_failed' && enableAutoRetry) {
        const retryCount = retryCountRef.current.get(scanMessage.scanId) || 0;
        if (retryCount < maxRetryAttempts) {
          retryCountRef.current.set(scanMessage.scanId, retryCount + 1);
          logger.info(`Retrying scan ${scanMessage.scanId}`, { attempt: retryCount + 1 });
          
          setTimeout(() => {
            // Use ref to avoid circular dependency
            if (baseRef.current?.sendMessage) {
              baseRef.current.sendMessage({
                type: 'retry_scan',
                scanId: scanMessage.scanId
              });
            }
          }, retryDelay * Math.pow(2, retryCount));
        }
      }
    }
  }, [onScanUpdate, onScanCompleted, onScanFailed, enableAutoRetry, maxRetryAttempts, retryDelay]);

  // Handle errors with toast notifications
  const handleError = useCallback((error: Error) => {
    logger.error('Scan WebSocket error', error);
    
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

  // Update base reference for retry logic
  useEffect(() => {
    baseRef.current = base;
  }, [base]);

  // Subscribe to a scan's updates
  const subscribeToScan = useCallback((scanId: string) => {
    if (!scanId) {
      logger.warn('Attempted to subscribe to scan with empty ID');
      return () => { };
    }

    if (subscribedScans.current.has(scanId)) {
      logger.debug('Already subscribed to scan', { scanId });
      return () => {
        // Direct unsubscribe logic to avoid circular dependency
        if (!scanId) return;
        
        if (subscribedScans.current.has(scanId)) {
          subscribedScans.current.delete(scanId);
          retryCountRef.current.delete(scanId);
          
          const success = base.sendMessage({
            type: 'unsubscribe_scan',
            payload: { scanId },
            timestamp: new Date().toISOString(),
            id: crypto.randomUUID()
          });
          
          if (success) {
            logger.info('Unsubscribed from scan', { scanId });
          }
        }
      };
    }

    subscribedScans.current.add(scanId);
    retryCountRef.current.delete(scanId); // Reset retry count

    const success = base.sendMessage({
      type: 'subscribe_scan',
      payload: { scanId },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });

    if (success) {
      logger.info('Subscribed to scan', { scanId });
    }

    // Return unsubscribe function
    return () => {
      // Direct unsubscribe logic to avoid circular dependency
      if (!scanId) return;
      
      if (subscribedScans.current.has(scanId)) {
        subscribedScans.current.delete(scanId);
        retryCountRef.current.delete(scanId);
        
        const success = base.sendMessage({
          type: 'unsubscribe_scan',
          payload: { scanId },
          timestamp: new Date().toISOString(),
          id: crypto.randomUUID()
        });
        
        if (success) {
          logger.info('Unsubscribed from scan', { scanId });
        }
      }
    };
  }, [base, subscribedScans, retryCountRef]);

  // Unsubscribe from a scan's updates
  const unsubscribeFromScan = useCallback((scanId: string) => {
    if (!scanId) return;

    if (!subscribedScans.current.has(scanId)) {
      logger.debug('Not subscribed to scan', { scanId });
      return;
    }

    subscribedScans.current.delete(scanId);
    retryCountRef.current.delete(scanId);

    const success = base.sendMessage({
      type: 'unsubscribe_scan',
      payload: { scanId },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });

    if (success) {
      logger.info('Unsubscribed from scan', { scanId });
    }
  }, [base, subscribedScans, retryCountRef]);

  // Get subscription status
  const isSubscribed = useCallback((scanId: string) => {
    return subscribedScans.current.has(scanId);
  }, []);

  // Get retry count
  const getRetryCount = useCallback((scanId: string) => {
    return retryCountRef.current.get(scanId) || 0;
  }, []);

  // Unsubscribe from all scans
  const unsubscribeFromAll = useCallback(() => {
    const scanIds = Array.from(subscribedScans.current);
    scanIds.forEach(scanId => {
      if (!scanId) return;
      
      if (subscribedScans.current.has(scanId)) {
        subscribedScans.current.delete(scanId);
        retryCountRef.current.delete(scanId);
        
        const success = base.sendMessage({
          type: 'unsubscribe_scan',
          payload: { scanId },
          timestamp: new Date().toISOString(),
          id: crypto.randomUUID()
        });
        
        if (success) {
          logger.info('Unsubscribed from scan', { scanId });
        }
      }
    });
    logger.info('Unsubscribed from all scans', { count: scanIds.length });
  }, [base, subscribedScans, retryCountRef]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      unsubscribeFromAll();
    };
  }, [unsubscribeFromAll]);

  // Get subscribed scans count (callback to avoid ref access during render)
  const getSubscribedScans = useCallback(() => {
    return Array.from(subscribedScans.current);
  }, []);

  return {
    subscribeToScan,
    unsubscribeFromScan,
    unsubscribeFromAll,
    isSubscribed,
    getRetryCount,
    isConnected: base.isConnected,
    connectionStatus: base.connectionStatus,
    sendMessage: base.sendMessage,
    reconnect: base.reconnect,
    getSubscribedScans
  };
}
