import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { notificationService, Notification } from '../services/notificationService';
import { useToast } from '@/components/ui/use-toast';

interface NotificationContextType {
    notifications: Notification[];
    unreadCount: number;
    addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => Notification;
    removeNotification: (id: string) => void;
    markAsRead: (id: string) => void;
    markAllAsRead: () => void;
    clearAll: () => void;
    // Helper methods
    success: (title: string, message: string, options?: Partial<Notification>) => Notification;
    error: (title: string, message: string, options?: Partial<Notification>) => Notification;
    warning: (title: string, message: string, options?: Partial<Notification>) => Notification;
    info: (title: string, message: string, options?: Partial<Notification>) => Notification;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const { toast } = useToast();

    // Subscribe to notification service
    useEffect(() => {
        const unsubscribe = notificationService.subscribe((updatedNotifications) => {
            setNotifications(updatedNotifications);
        });

        return unsubscribe;
    }, []);

    // Show toast for new notifications
    useEffect(() => {
        if (notifications.length > 0) {
            const latestNotification = notifications[0];
            if (!latestNotification.read) {
                // Show toast for important notifications
                if (latestNotification.type !== 'info') {
                    toast({
                        title: latestNotification.title,
                        description: latestNotification.message,
                        variant: latestNotification.type === 'error' ? 'destructive' : 'default',
                    });
                }
            }
        }
    }, [notifications, toast]);

    const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
        return notificationService.add(notification);
    }, []);

    const removeNotification = useCallback((id: string) => {
        notificationService.remove(id);
    }, []);

    const markAsRead = useCallback((id: string) => {
        notificationService.markAsRead(id);
    }, []);

    const markAllAsRead = useCallback(() => {
        notificationService.markAllAsRead();
    }, []);

    const clearAll = useCallback(() => {
        notificationService.clearAll();
    }, []);

    const success = useCallback((title: string, message: string, options?: Partial<Notification>) => {
        return notificationService.success(title, message, options);
    }, []);

    const error = useCallback((title: string, message: string, options?: Partial<Notification>) => {
        return notificationService.error(title, message, options);
    }, []);

    const warning = useCallback((title: string, message: string, options?: Partial<Notification>) => {
        return notificationService.warning(title, message, options);
    }, []);

    const info = useCallback((title: string, message: string, options?: Partial<Notification>) => {
        return notificationService.info(title, message, options);
    }, []);

    const unreadCount = notifications.filter(n => !n.read).length;

    const value: NotificationContextType = {
        notifications,
        unreadCount,
        addNotification,
        removeNotification,
        markAsRead,
        markAllAsRead,
        clearAll,
        success,
        error,
        warning,
        info
    };

    return (
        <NotificationContext.Provider value={value}>
            {children}
        </NotificationContext.Provider>
    );
}

export function useNotifications() {
    const context = useContext(NotificationContext);
    if (!context) {
        throw new Error('useNotifications must be used within a NotificationProvider');
    }
    return context;
}
