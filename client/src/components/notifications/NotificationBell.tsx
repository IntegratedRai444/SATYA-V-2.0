import React, { useState } from 'react';
import { Bell, Check, AlertCircle, Info, CheckCircle, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { useRealtime } from '@/contexts/RealtimeContext';
import { useNotifications } from '@/contexts/NotificationContext';

const NotificationIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'error':
      return <AlertCircle className="h-4 w-4 text-destructive" />;
    case 'warning':
      return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    case 'success':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    default:
      return <Info className="h-4 w-4 text-blue-500" />;
  }
};

export const NotificationBell = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { notifications: realtimeNotifications, unreadCount: realtimeUnreadCount, markAsRead: realtimeMarkAsRead, clearAll: realtimeClearAll } = useRealtime();
  const { notifications: localNotifications, unreadCount: localUnreadCount, markAsRead: localMarkAsRead, clearAll: localClearAll } = useNotifications();

  // Merge notifications from both sources
  const notifications = [...realtimeNotifications, ...localNotifications];
  const unreadCount = realtimeUnreadCount + localUnreadCount;

  const markAsRead = (id: string) => {
    realtimeMarkAsRead(id);
    localMarkAsRead(id);
  };

  const clearAll = () => {
    realtimeClearAll();
    localClearAll();
  };

  const handleMarkAsRead = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    markAsRead(id);
  };

  const handleClearAll = (e: React.MouseEvent) => {
    e.stopPropagation();
    clearAll();
  };

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="icon"
        className="relative"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Bell className="h-5 w-5" />
        {unreadCount > 0 && (
          <Badge
            variant="destructive"
            className="absolute -right-1 -top-1 h-5 w-5 rounded-full p-0 flex items-center justify-center"
          >
            {unreadCount > 9 ? '9+' : unreadCount}
          </Badge>
        )}
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 rounded-md border bg-popover shadow-lg z-50">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <h3 className="font-medium">Notifications</h3>
            <div className="flex items-center space-x-2">
              {notifications.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={handleClearAll}
                >
                  Clear all
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => setIsOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {notifications.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No new notifications
            </div>
          ) : (
            <div className="max-h-96 overflow-y-auto">
              <div className="divide-y">
                {notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={cn(
                      'p-4 hover:bg-accent/50 transition-colors cursor-pointer',
                      !notification.read && 'bg-accent/20'
                    )}
                    onClick={() => {
                      markAsRead(notification.id);
                      // Handle notification click (e.g., navigate to relevant page)
                    }}
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">
                        <NotificationIcon type={notification.type} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{notification.title}</h4>
                          <div className="flex items-center text-xs text-muted-foreground">
                            {new Date(notification.timestamp).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                            {!notification.read && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="ml-2 h-5 w-5"
                                onClick={(e) => handleMarkAsRead(e, notification.id)}
                              >
                                <Check className="h-3 w-3" />
                              </Button>
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {notification.message}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
