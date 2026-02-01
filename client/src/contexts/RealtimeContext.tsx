import React, { createContext, useState, useEffect, useRef, useCallback } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { webSocketService } from '@/services/websocket';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import type { WebSocketMessage, ScanUpdateMessage, SystemAlertMessage } from '@/types/websocket';

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  read?: boolean;
  autoClose?: boolean;
  duration?: number;
}

export interface RealtimeContextType {
  notifications: Notification[];
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  lastUpdate: Date | null;
  activeScans: Record<string, {
    id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress: number;
    fileName?: string;
    error?: string;
    timestamp: Date;
  }>;
  reconnectAttempt: number;
  maxReconnectAttempts: number;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  clearAll: () => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  reconnect: () => void;
  subscribeToScan: (scanId: string) => () => void;
  unsubscribeFromScan: (scanId: string) => void;
  sendMessage: (message: unknown) => boolean;
  unreadCount: number;
}

const RealtimeContext = createContext<RealtimeContextType | undefined>(undefined);

// Export the context separately for Fast Refresh compatibility
export { RealtimeContext };

export const RealtimeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user } = useSupabaseAuth();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [activeScans, setActiveScans] = useState<RealtimeContextType['activeScans']>({});
  const [reconnectAttempt, setReconnectAttempt] = useState<number>(0);
  const [maxReconnectAttempts] = useState<number>(5);
  const { toast } = useToast();
  
  // User is authenticated if we have a user object
  const isAuthenticated = !!user;
  const subscriptions = useRef<Record<string, () => void>>({});

  // Add a new notification
  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `${Date.now()} -${Math.random().toString(36).substr(2, 9)} `,
      timestamp: new Date(),
      read: false,
    };

    setNotifications((prev: Notification[]) => [newNotification, ...prev].slice(0, 50)); // Keep only the 50 most recent
    setLastUpdate(new Date());

    // Show toast for important notifications
    if (notification.type !== 'info') {
      toast({
        title: notification.title,
        description: notification.message,
        variant: notification.type === 'error' ? 'destructive' : 'default',
      });
    }
  }, [toast]);

  // Mark notification as read
  const markAsRead = useCallback((id: string) => {
    setNotifications((prev: Notification[]) =>
      prev.map((notif: Notification) =>
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  }, []);

  // Mark all notifications as read
  const markAllAsRead = useCallback(() => {
    setNotifications((prev: Notification[]) =>
      prev.map((notif: Notification) => ({
        ...notif,
        read: true
      }))
    );
  }, []);

  // Clear all notifications
  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  // Remove a specific notification
  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  }, []);

  // Reconnect WebSocket
  const reconnect = useCallback(() => {
    if (webSocketService) {
      webSocketService.disconnect();
      // Connection will be re-established by the useEffect
    }
  }, []);

  // Send a message through WebSocket
  const sendMessage = useCallback((message: unknown) => {
    if (webSocketService) {
      webSocketService.send(message as Record<string, unknown>);
      return true;
    }
    return false;
  }, []);

  // Subscribe to a scan's updates
  const subscribeToScan = useCallback((scanId: string): (() => void) => {
    if (!scanId || subscriptions.current[scanId]) return (() => {});

    // Subscribe to WebSocket channel
    webSocketService.subscribeToScan(scanId);

    // Track the subscription for cleanup
    const unsubscribe = webSocketService.subscribe((message: WebSocketMessage) => {
      if (message.type === 'scan_update' && message.payload.scanId === scanId) {
        const scanMessage = message as ScanUpdateMessage;
        setActiveScans((prev: RealtimeContextType['activeScans']) => ({
          ...prev,
          [scanId]: {
            id: scanId,
            status: scanMessage.payload.status,
            progress: scanMessage.payload.progress || 0,
            fileName: scanMessage.payload.fileName,
            error: scanMessage.payload.error,
            timestamp: new Date(scanMessage.timestamp)
          }
        }));

        // Add notification for important updates
        if (scanMessage.payload.status === 'completed' || scanMessage.payload.status === 'failed') {
          addNotification({
            type: scanMessage.payload.status === 'completed' ? 'success' : 'error',
            title: `Scan ${scanMessage.payload.status} `,
            message: scanMessage.payload.fileName
              ? `File "${scanMessage.payload.fileName}" scan ${scanMessage.payload.status} `
              : `Scan ${scanMessage.payload.status} `
          });
        }
      }
    });

    // Store the unsubscribe function
    subscriptions.current[scanId] = unsubscribe;

    return () => {
      unsubscribe();
      delete subscriptions.current[scanId];
      webSocketService.unsubscribeFromScan(scanId);
    };
  }, [addNotification]);

  // Unsubscribe from a scan's updates
  const unsubscribeFromScan = useCallback((scanId: string) => {
    if (subscriptions.current[scanId]) {
      subscriptions.current[scanId]();
      delete subscriptions.current[scanId];
      webSocketService.unsubscribeFromScan(scanId);
    }
  }, []);

  // Set up WebSocket listeners
  useEffect(() => {
    if (!isAuthenticated) return;

    const handleConnectionStatus = (status: 'connected' | 'disconnected') => {
      setConnectionStatus(status);
      if (status === 'connected') {
        addNotification({
          type: 'success',
          title: 'Connected',
          message: 'Real-time updates are now active',
        });
      } else {
        addNotification({
          type: 'warning',
          title: 'Connection Lost',
          message: 'Real-time updates are not available. Trying to reconnect...',
        });
      }
    };

    // Subscribe to WebSocket events
    const unsubscribeMessage = webSocketService.subscribe((data: WebSocketMessage) => {
      // Handle system alerts
      if (data.type === 'system_alert') {
        const alert = data as SystemAlertMessage;
        addNotification({
          type: alert.payload.severity === 'critical' ? 'error' : (alert.payload.severity as NotificationType) || 'warning',
          title: alert.payload.title,
          message: alert.payload.message,
          // Note: WebSocket action has callback/url, not onClick - would need adapter if actions are needed
        });
      }
    });

    const unsubscribeConnected = webSocketService.on('connected', () => {
      handleConnectionStatus('connected');
    });

    const unsubscribeDisconnected = webSocketService.on('disconnected', () => {
      handleConnectionStatus('disconnected');
    });

    const unsubscribeError = webSocketService.on('error', (...args: unknown[]) => {
      const error = args[0] as Error;
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: error.message || 'An error occurred with the real-time connection',
      });
    });

    const unsubscribeReconnecting = webSocketService.on('reconnecting', (...args: unknown[]) => {
      const data = args[0] as { attempt: number; maxAttempts: number };
      setConnectionStatus('connecting');
      setReconnectAttempt(data.attempt);
      addNotification({
        type: 'info',
        title: 'Reconnecting',
        message: `Attempting to reconnect(${data.attempt} / ${data.maxAttempts})...`,
      });
    });

    // Initial connection
    webSocketService.connect().catch((error: Error) => {
      console.error('WebSocket connection error:', error);
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: `Failed to connect to real - time service: ${error.message} `,
      });
    });

    // Clean up
    return () => {
      unsubscribeMessage();
      unsubscribeConnected();
      unsubscribeDisconnected();
      unsubscribeError();
      unsubscribeReconnecting();

      // Clean up all scan subscriptions
      Object.values(subscriptions.current).forEach(unsubscribe => unsubscribe());
      subscriptions.current = {};

      // Disconnect WebSocket if needed
      if (webSocketService) {
        webSocketService.disconnect();
      }
    };
  }, [isAuthenticated, addNotification]);

  // Calculate unread count
  const unreadCount = notifications.filter(n => !n.read).length;

  const contextValue: RealtimeContextType = {
    // Connection state
    connectionStatus,
    lastUpdate,
    reconnectAttempt,
    maxReconnectAttempts,

    // Notifications
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    markAllAsRead,
    clearNotifications: clearAll,
    clearAll,
    reconnect,

    // WebSocket methods
    subscribeToScan: (scanId: string) => {
      const unsubscribe = subscribeToScan(scanId);
      return unsubscribe || (() => {});
    },
    unsubscribeFromScan,
    sendMessage,

    // Scan progress
    activeScans,
    unreadCount,
  };

  return (
    <RealtimeContext.Provider value={contextValue}>
      {children}
    </RealtimeContext.Provider>
  );
};
