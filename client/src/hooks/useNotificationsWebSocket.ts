import { useCallback, useEffect, useState } from 'react';
import { useBaseWebSocket, WebSocketMessage } from './useBaseWebSocket';
import logger from '../lib/logger';
import { useToast } from '@/components/ui/use-toast';

export interface NotificationMessage extends WebSocketMessage {
  type: 'notification';
  payload: {
    id: string;
    title: string;
    message: string;
    type: 'info' | 'warning' | 'error' | 'success';
    priority?: 'low' | 'normal' | 'high' | 'urgent';
    category?: 'system' | 'analysis' | 'security' | 'update';
    timestamp: string;
    read?: boolean;
    dismissed?: boolean;
    duration?: number;
    action?: {
      label: string;
      callback?: string;
      url?: string;
    };
  };
}

interface UseNotificationsWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  enableToast?: boolean;
  maxStoredNotifications?: number;
  onNotification?: (notification: NotificationMessage['payload']) => void;
  onNotificationRead?: (notificationId: string) => void;
  onNotificationDismissed?: (notificationId: string) => void;
}

export function useNotificationsWebSocket(options: UseNotificationsWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 5000,
    enableToast = true,
    maxStoredNotifications = 50,
    onNotification,
    onNotificationRead,
    onNotificationDismissed
  } = options;

  const [notifications, setNotifications] = useState<Map<string, NotificationMessage['payload']>>(new Map());
  const { toast } = useToast();

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      if (message.type === 'notification') {
        const notification = message as NotificationMessage;
        logger.debug('Notification received', { id: notification.payload.id, type: notification.payload.type });

        // Store notification
        setNotifications(prev => {
          const newMap = new Map(prev);
          newMap.set(notification.payload.id, notification.payload);
          
          // Limit stored notifications
          if (newMap.size > maxStoredNotifications) {
            const oldestId = Array.from(newMap.keys())[0];
            if (oldestId) {
              newMap.delete(oldestId);
            }
          }
          
          return newMap;
        });
        
        // Show toast notification if enabled
        if (enableToast) {
          toast({
            title: notification.payload.title,
            description: notification.payload.message,
            variant: notification.payload.type === 'error' ? 'destructive' : 
                   notification.payload.type === 'warning' ? 'default' : 
                   'default',
            duration: notification.payload.duration || 5000
          });
        }
        
        onNotification?.(notification.payload);
      } else if (message.type === 'notification_read') {
        const notificationId = (message.payload as { notificationId?: string })?.notificationId;
        if (notificationId) {
          logger.debug('Notification read', { notificationId });
          onNotificationRead?.(notificationId);
          
          // Update notification read status
          setNotifications(prev => {
            const newMap = new Map(prev);
            const existing = newMap.get(notificationId);
            if (existing) {
              newMap.set(notificationId, { ...existing, read: true });
            }
            return newMap;
          });
        }
      } else if (message.type === 'notification_dismissed') {
        const notificationId = (message.payload as { notificationId?: string })?.notificationId;
        if (notificationId) {
          logger.debug('Notification dismissed', { notificationId });
          onNotificationDismissed?.(notificationId);
          
          // Remove notification from storage
          setNotifications(prev => {
            const newMap = new Map(prev);
            newMap.delete(notificationId);
            return newMap;
          });
        }
      }
    } catch (error) {
      logger.error('Error handling notification message', error instanceof Error ? error : new Error(String(error)));
    }
  }, [enableToast, maxStoredNotifications, onNotification, onNotificationRead, onNotificationDismissed, toast]);

  // Use base WebSocket with notifications-specific configuration
  const base = useBaseWebSocket({
    autoConnect,
    reconnectAttempts,
    reconnectInterval,
    onMessage: handleMessage,
  });

  // Subscribe to notifications channel on connection
  const subscribeToNotifications = useCallback(() => {
    if (!base.isConnected) {
      logger.warn('WebSocket not connected, cannot subscribe to notifications');
      return false;
    }
    
    return base.sendMessage({
      type: 'subscribe',
      payload: { channel: 'notifications' },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [base]);

  // Subscribe automatically when connected
  useEffect(() => {
    if (base.isConnected) {
      subscribeToNotifications();
    }
  }, [base.isConnected, subscribeToNotifications]);

  // Dismiss notification
  const dismissNotification = useCallback((notificationId: string) => {
    if (!base.isConnected) {
      logger.warn('WebSocket not connected, cannot dismiss notification');
      return false;
    }
    
    return base.sendMessage({
      type: 'notification_dismissed',
      payload: { notificationId },
      timestamp: new Date().toISOString(),
      id: crypto.randomUUID()
    });
  }, [base]);

  // Get notification by ID
  const getNotification = useCallback((notificationId: string) => {
    return notifications.get(notificationId);
  }, [notifications]);

  // Get all notifications
  const getAllNotifications = useCallback(() => {
    return Array.from(notifications.values()).sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [notifications]);

  // Get unread notifications
  const getUnreadNotifications = useCallback(() => {
    return Array.from(notifications.values())
      .filter(n => !n.read)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [notifications]);

  // Get unread count
  const getUnreadCount = useCallback(() => {
    return Array.from(notifications.values()).filter(n => !n.read).length;
  }, [notifications]);

  // Clear all notifications
  const clearAllNotifications = useCallback(() => {
    setNotifications(new Map());
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => {
      const newMap = new Map();
      prev.forEach((notification, id) => {
        newMap.set(id, { ...notification, read: true });
      });
      return newMap;
    });
  }, []);

  return {
    isConnected: base.isConnected,
    connectionError: base.connectionError,
    connectionStatus: base.connectionStatus,
    connect: base.connect,
    disconnect: base.disconnect,
    sendMessage: base.sendMessage,
    subscribeToNotifications,
    dismissNotification,
    getNotification,
    getAllNotifications,
    getUnreadNotifications,
    getUnreadCount,
    clearAllNotifications,
    markAllAsRead,
    notifications: Array.from(notifications.values()),
    reconnect: base.reconnect,
  };
}
