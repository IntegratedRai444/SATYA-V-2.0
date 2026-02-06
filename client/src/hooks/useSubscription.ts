import { useCallback, useEffect, useRef, useState } from 'react';
import type { WebSocketMessage } from '@/types/websocket';

/**
 * Generic WebSocket subscription hook
 * Eliminates duplication across useWebSocket, useDashboardWebSocket, useScanWebSocket, and RealtimeContext
 * 
 * @param baseWebSocket - The base WebSocket instance to use for communication
 * @param subscriptionId - Unique identifier for this subscription (e.g., jobId, scanId, channel)
 * @param subscribeType - Message type for subscribing (e.g., 'subscribe_job', 'subscribe_scan')
 * @param unsubscribeType - Message type for unsubscribing (e.g., 'unsubscribe_job', 'unsubscribe_scan')
 * @param messageFilter - Function to filter relevant messages
 * @param onUpdate - Callback function when relevant message is received
 * @param payload - Optional payload to include with subscription
 * @param autoSubscribe - Whether to automatically subscribe on mount
 */
export function useSubscription<TData = unknown>(
    baseWebSocket: {
        sendMessage: (message: unknown) => boolean;
        subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
        isConnected: boolean;
    },
    subscriptionId: string | null,
    subscribeType: string,
    unsubscribeType: string,
    messageFilter: (message: WebSocketMessage) => boolean,
    onUpdate: (data: TData) => void,
    payload: Record<string, unknown> = {},
    autoSubscribe = true
) {
    const subscriptionRef = useRef<(() => void) | null>(null);
    const [isSubscribed, setIsSubscribed] = useState(false);

    // Subscribe function
    const subscribe = useCallback(() => {
        if (!baseWebSocket || !subscriptionId || isSubscribed) {
            return;
        }

        // Send subscribe message
        baseWebSocket.sendMessage({
            type: subscribeType,
            [getIdKey(subscribeType)]: subscriptionId,
            ...payload
        });

        // Set up message handler
        const unsubscribe = baseWebSocket.subscribe((message: WebSocketMessage) => {
            if (messageFilter(message)) {
                const messageData = (message as unknown as Record<string, unknown>).payload || (message as unknown as Record<string, unknown>).data;
                onUpdate(messageData as TData);
            }
        });

        subscriptionRef.current = unsubscribe;
        setIsSubscribed(true);
    }, [baseWebSocket, subscriptionId, subscribeType, messageFilter, onUpdate, payload, isSubscribed]);

    // Unsubscribe function
    const unsubscribe = useCallback(() => {
        if (!baseWebSocket || !subscriptionId || !isSubscribed) {
            return;
        }

        // Send unsubscribe message
        baseWebSocket.sendMessage({
            type: unsubscribeType,
            [getIdKey(unsubscribeType)]: subscriptionId
        });

        // Clean up message handler
        if (subscriptionRef.current) {
            subscriptionRef.current();
            subscriptionRef.current = null;
        }

        setIsSubscribed(false);
    }, [baseWebSocket, subscriptionId, unsubscribeType, isSubscribed]);

    // Auto-subscribe on mount if enabled
    useEffect(() => {
        if (autoSubscribe && subscriptionId && baseWebSocket?.isConnected) {
            // Defer the subscribe call to avoid synchronous setState
            const timeoutId = setTimeout(() => {
                subscribe();
            }, 0);
            
            return () => {
                clearTimeout(timeoutId);
            };
        }

        return () => {
            unsubscribe();
        };
    }, [autoSubscribe, subscriptionId, baseWebSocket?.isConnected, subscribe, unsubscribe]);

    return {
        subscribe,
        unsubscribe,
        isSubscribed
    };
}

/**
 * Helper to extract the ID key from subscription type
 * e.g., 'subscribe_job' -> 'jobId', 'subscribe_scan' -> 'scanId'
 */
function getIdKey(type: string): string {
    const match = type.match(/subscribe_(\w+)/);
    if (match) {
        return `${match[1]}Id`;
    }
    return 'id';
}

/**
 * Specialized hook for job subscriptions
 */
export function useJobSubscription(
    baseWebSocket: {
        sendMessage: (message: unknown) => boolean;
        subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
        isConnected: boolean;
    },
    jobId: string | null,
    onUpdate: (data: unknown) => void
) {
    return useSubscription(
        baseWebSocket,
        jobId,
        'subscribe_job',
        'unsubscribe_job',
        (message) => (message as unknown as { type: string; data?: { jobId?: string } }).type === 'job_update' && (message as unknown as { data?: { jobId?: string } }).data?.jobId === jobId,
        onUpdate
    );
}

/**
 * Specialized hook for scan subscriptions
 */
export function useScanSubscription(
    baseWebSocket: {
        sendMessage: (message: unknown) => boolean;
        subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
        isConnected: boolean;
    },
    scanId: string | null,
    onUpdate: (data: unknown) => void
) {
    return useSubscription(
        baseWebSocket,
        scanId,
        'subscribe_scan',
        'unsubscribe_scan',
        (message) => (message as unknown as { type: string; payload?: { scanId?: string } }).type === 'scan_update' && (message as unknown as { payload?: { scanId?: string } }).payload?.scanId === scanId,
        onUpdate
    );
}

/**
 * Specialized hook for channel subscriptions (dashboard, etc.)
 */
export function useChannelSubscription(
    baseWebSocket: {
        sendMessage: (message: unknown) => boolean;
        subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
        isConnected: boolean;
    },
    channel: string | null,
    onUpdate: (data: unknown) => void
) {
    return useSubscription(
        baseWebSocket,
        channel,
        'subscribe',
        'unsubscribe',
        (message) => {
            const channelTypes = ['stats', 'analytics', 'activity', 'metrics', 'all'];
            return channelTypes.includes((message as unknown as { type: string }).type);
        },
        onUpdate,
        { payload: { channel } }
    );
}
