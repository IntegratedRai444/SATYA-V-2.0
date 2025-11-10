import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { webSocketService, WebSocketMessage, NotificationMessage, ScanUpdateMessage, SystemAlertMessage } from '@/services/websocket';
import { useToast } from '@/components/ui/use-toast';

type WebSocketContextType = {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  notifications: NotificationMessage[];
  scanUpdates: { [key: string]: ScanUpdateMessage };
  systemAlerts: SystemAlertMessage[];
  subscribeToScan: (scanId: string) => void;
  unsubscribeFromScan: (scanId: string) => void;
  clearNotification: (id: string) => void;
  clearAlert: (id: string) => void;
};

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [notifications, setNotifications] = useState<NotificationMessage[]>([]);
  const [scanUpdates, setScanUpdates] = useState<{ [key: string]: ScanUpdateMessage }>({});
  const [systemAlerts, setSystemAlerts] = useState<SystemAlertMessage[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    // Connect to WebSocket
    const connectWebSocket = async () => {
      try {
        await webSocketService.connect();
      } catch (error) {
        console.error('WebSocket connection error:', error);
        toast({
          title: 'Connection Error',
          description: 'Failed to connect to real-time service',
          variant: 'destructive',
        });
      }
    };

    // Set up event handlers
    const onConnected = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    const onDisconnected = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    const onMessage = (message: WebSocketMessage) => {
      setLastMessage(message);
      
      switch (message.type) {
        case 'notification':
          setNotifications(prev => [message as NotificationMessage, ...prev].slice(0, 50));
          break;
          
        case 'scan_update':
          const update = message as ScanUpdateMessage;
          setScanUpdates(prev => ({
            ...prev,
            [update.scanId]: update
          }));
          
          // Show toast for important updates
          if (update.status === 'completed' || update.status === 'failed') {
            toast({
              title: `Scan ${update.status}`,
              description: update.fileName ? `${update.fileName} processing ${update.status}` : `Scan ${update.status}`,
              variant: update.status === 'completed' ? 'default' : 'destructive'
            });
          }
          break;
          
        case 'system_alert':
          setSystemAlerts(prev => [message as SystemAlertMessage, ...prev].slice(0, 20));
          toast({
            title: 'System Alert',
            description: message.message,
            variant: 'destructive',
          });
          break;
      }
    };

    // Subscribe to WebSocket events
    webSocketService.on('connected', onConnected);
    webSocketService.on('disconnected', onDisconnected);
    webSocketService.on('message', onMessage);

    // Connect to WebSocket
    connectWebSocket();

    // Cleanup
    return () => {
      webSocketService.off('connected', onConnected);
      webSocketService.off('disconnected', onDisconnected);
      webSocketService.off('message', onMessage);
      webSocketService.disconnect();
    };
  }, [toast]);

  const subscribeToScan = (scanId: string) => {
    webSocketService.subscribeToScan(scanId);
  };

  const unsubscribeFromScan = (scanId: string) => {
    webSocketService.unsubscribeFromScan(scanId);
  };

  const clearNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const clearAlert = (id: string) => {
    setSystemAlerts(prev => prev.filter(a => a.id !== id));
  };

  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        lastMessage,
        notifications,
        scanUpdates,
        systemAlerts,
        subscribeToScan,
        unsubscribeFromScan,
        clearNotification,
        clearAlert,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};
