/**
 * Centralized Notification Service
 * Eliminates notification duplication across:
 * - AppContext.tsx
 * - RealtimeContext.tsx
 * - SupabaseAuthProvider.tsx
 * - useNotifications.ts hook
 */

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

export interface Notification {
    id: string;
    type: NotificationType;
    title: string;
    message: string;
    timestamp: Date;
    read: boolean;
    duration?: number;
    data?: any;
    action?: {
        label: string;
        onClick: () => void;
    };
}

type NotificationListener = (notifications: Notification[]) => void;

class NotificationService {
    private notifications: Notification[] = [];
    private listeners: Set<NotificationListener> = new Set();
    private maxNotifications = 50;

    /**
     * Add a new notification
     */
    add(notification: Omit<Notification, 'id' | 'timestamp' | 'read'>): Notification {
        const newNotification: Notification = {
            ...notification,
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date(),
            read: false
        };

        this.notifications = [newNotification, ...this.notifications].slice(0, this.maxNotifications);
        this.notifyListeners();

        // Auto-remove after duration if specified
        if (notification.duration) {
            setTimeout(() => {
                this.remove(newNotification.id);
            }, notification.duration);
        }

        return newNotification;
    }

    /**
     * Remove a notification by ID
     */
    remove(id: string): void {
        this.notifications = this.notifications.filter(n => n.id !== id);
        this.notifyListeners();
    }

    /**
     * Mark notification as read
     */
    markAsRead(id: string): void {
        this.notifications = this.notifications.map(n =>
            n.id === id ? { ...n, read: true } : n
        );
        this.notifyListeners();
    }

    /**
     * Mark all notifications as read
     */
    markAllAsRead(): void {
        this.notifications = this.notifications.map(n => ({ ...n, read: true }));
        this.notifyListeners();
    }

    /**
     * Clear all notifications
     */
    clearAll(): void {
        this.notifications = [];
        this.notifyListeners();
    }

    /**
     * Get all notifications
     */
    getAll(): Notification[] {
        return [...this.notifications];
    }

    /**
     * Get unread count
     */
    getUnreadCount(): number {
        return this.notifications.filter(n => !n.read).length;
    }

    /**
     * Subscribe to notification changes
     */
    subscribe(listener: NotificationListener): () => void {
        this.listeners.add(listener);
        // Immediately notify with current state
        listener(this.getAll());

        // Return unsubscribe function
        return () => {
            this.listeners.delete(listener);
        };
    }

    /**
     * Notify all listeners of changes
     */
    private notifyListeners(): void {
        const notifications = this.getAll();
        this.listeners.forEach(listener => {
            try {
                listener(notifications);
            } catch (error) {
                if (import.meta.env.DEV) {
                    console.error('Error in notification listener:', error);
                }
            }
        });
    }

    /**
     * Helper methods for common notification types
     */
    success(title: string, message: string, options?: Partial<Notification>): Notification {
        return this.add({ type: 'success', title, message, ...options });
    }

    error(title: string, message: string, options?: Partial<Notification>): Notification {
        return this.add({ type: 'error', title, message, ...options });
    }

    warning(title: string, message: string, options?: Partial<Notification>): Notification {
        return this.add({ type: 'warning', title, message, ...options });
    }

    info(title: string, message: string, options?: Partial<Notification>): Notification {
        return this.add({ type: 'info', title, message, ...options });
    }
}

// Export singleton instance
export const notificationService = new NotificationService();

// Export class for testing
export { NotificationService };
