import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { 
  webSocketService, 
  WebSocketMessage, 
  ScanUpdateMessage, 
  NotificationMessage, 
  SystemAlertMessage 
} from '@/services/websocket';
import { useToast } from '@/components/ui/use-toast';
import { useAuth } from './AuthContext';

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  data?: any;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface RealtimeContextType {
  // Connection state
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  lastUpdate: Date | null;
  
  // Notifications
  notifications: Notification[];
  unreadCount: number;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
  
  // WebSocket methods
  subscribeToScan: (scanId: string) => void;
  unsubscribeFromScan: (scanId: string) => void;
  sendMessage: <T = any>(message: T) => void;
  
  // Scan progress tracking
  activeScans: Record<string, {
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress: number;
    fileName?: string;
    error?: string;
    timestamp: Date;
  }>;
}

const RealtimeContext = createContext<RealtimeContextType | undefined>(undefined);

export const RealtimeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [activeScans, setActiveScans] = useState<RealtimeContextType['activeScans']>({});
  const { toast } = useToast();
  const { isAuthenticated } = useAuth();
  const subscriptions = useRef<Record<string, () => void>>({});

  // Add a new notification
  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      read: false,
    };

    setNotifications(prev => [newNotification, ...prev].slice(0, 50)); // Keep only the 50 most recent
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
    setNotifications(prev =>
      prev.map(notif =>
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  }, []);

  // Mark all notifications as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => 
      prev.map(notif => ({
        ...notif,
        read: true
      }))
    );
  }, []);

  // Clear all notifications
  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  // Send a message through WebSocket
  const sendMessage = useCallback(<T = any>(message: T) => {
    webSocketService.send(message);
  }, []);

  // Subscribe to a scan's updates
  const subscribeToScan = useCallback((scanId: string) => {
    if (!scanId || subscriptions.current[scanId]) return;
    
    // Subscribe to WebSocket channel
    webSocketService.subscribeToScan(scanId);
    
    // Track the subscription for cleanup
    const unsubscribe = webSocketService.subscribe((message: WebSocketMessage) => {
      if (message.type === 'scan_update' && message.scanId === scanId) {
        const scanMessage = message as ScanUpdateMessage;
        setActiveScans(prev => ({
          ...prev,
          [scanId]: {
            status: scanMessage.status,
            progress: scanMessage.progress || 0,
            fileName: scanMessage.fileName,
            error: scanMessage.error,
            timestamp: new Date(scanMessage.timestamp)
          }
        }));

        // Add notification for important updates
        if (scanMessage.status === 'completed' || scanMessage.status === 'failed') {
          addNotification({
            type: scanMessage.status === 'completed' ? 'success' : 'error',
            title: `Scan ${scanMessage.status}`,
            message: scanMessage.fileName 
              ? `File "${scanMessage.fileName}" scan ${scanMessage.status}`
              : `Scan ${scanMessage.status}`,
            data: { scanId }
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
          type: alert.severity === 'critical' ? 'error' : (alert.severity as NotificationType) || 'warning',
          title: alert.title,
          message: alert.message,
          action: alert.action ? {
            label: alert.action.label,
            onClick: alert.action.onClick
          } : undefined
        });
      }
    });

    const unsubscribeConnected = webSocketService.on('connected', () => {
      handleConnectionStatus('connected');
    });

    const unsubscribeDisconnected = webSocketService.on('disconnected', () => {
      handleConnectionStatus('disconnected');
    });

    const unsubscribeError = webSocketService.on('error', (error: Error) => {
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: error.message || 'An error occurred with the real-time connection',
      });
    });

    // Initial connection
    webSocketService.connect().catch((error: Error) => {
      console.error('WebSocket connection error:', error);
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: `Failed to connect to real-time service: ${error.message}`,
      });
    });

    // Clean up
    return () => {
      unsubscribeMessage();
      unsubscribeConnected();
      unsubscribeDisconnected();
      unsubscribeError();
      
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
    
    // Notifications
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    clearAll,
    
    // WebSocket methods
    subscribeToScan,
    unsubscribeFromScan,
    sendMessage,
    
    // Scan progress
    activeScans,
  };

  return (
    <RealtimeContext.Provider value={contextValue}>
      {children}
    </RealtimeContext.Provider>
  );
};

export const useRealtime = () => {
  const context = useContext(RealtimeContext);
  if (!context) {
    throw new Error('useRealtime must be used within a RealtimeProvider');
  }
  return context;
};
