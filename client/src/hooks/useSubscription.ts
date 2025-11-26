import { useCallback, useEffect, useRef } from 'react';
import type { WebSocketMessage } from './useBaseWebSocket';

/**
 * Generic WebSocket subscription hook
 * Eliminates duplication across useWebSocket, useDashboardWebSocket, useScanWebSocket, and RealtimeContext
 * 
 * @param baseWebSocket - The base WebSocket instance to use for communication
 * @param subscriptionId - Unique identifier for this subscription (e.g., jobId, scanId, channel)
 * @param subscribeType - Message type for subscribing (e.g., 'subscribe_job', 'subscribe_scan')
 * @param unsubscribeType - Message type for unsubscribing (e.g., 'unsubscribe_job', 'unsubscribe_scan')
 * @param messageFilter - Function to filter relevant messages
 * @param onUpdate - Callback when a relevant message is received
 * @param options - Additional options
 */
export function useSubscription<TData = any>(
    baseWebSocket: {
        sendMessage: (message: any) => boolean | void;
        subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
        isConnected: boolean;
    } | null,
    subscriptionId: string | null,
    subscribeType: string,
    unsubscribeType: string,
    messageFilter: (message: WebSocketMessage) => boolean,
    onUpdate: (data: TData) => void,
    options: {
        autoSubscribe?: boolean;
        payload?: Record<string, any>;
    } = {}
) {
    const { autoSubscribe = true, payload = {} } = options;
    const subscriptionRef = useRef<(() => void) | null>(null);
    const isSubscribedRef = useRef(false);

    // Subscribe function
    const subscribe = useCallback(() => {
        if (!baseWebSocket || !subscriptionId || isSubscribedRef.current) {
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
                onUpdate(message.data || message.payload);
            }
        });

        subscriptionRef.current = unsubscribe;
        isSubscribedRef.current = true;
    }, [baseWebSocket, subscriptionId, subscribeType, messageFilter, onUpdate, payload]);

    // Unsubscribe function
    const unsubscribe = useCallback(() => {
        if (!baseWebSocket || !subscriptionId || !isSubscribedRef.current) {
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

        isSubscribedRef.current = false;
    }, [baseWebSocket, subscriptionId, unsubscribeType]);

    // Auto-subscribe on mount if enabled
    useEffect(() => {
        if (autoSubscribe && subscriptionId && baseWebSocket?.isConnected) {
            subscribe();
        }

        return () => {
            unsubscribe();
        };
    }, [autoSubscribe, subscriptionId, baseWebSocket?.isConnected, subscribe, unsubscribe]);

    return {
        subscribe,
        unsubscribe,
        isSubscribed: isSubscribedRef.current
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
    baseWebSocket: any,
    jobId: string | null,
    onUpdate: (data: any) => void
) {
    return useSubscription(
        baseWebSocket,
        jobId,
        'subscribe_job',
        'unsubscribe_job',
        (message) => message.type === 'job_update' && message.data?.jobId === jobId,
        onUpdate
    );
}

/**
 * Specialized hook for scan subscriptions
 */
export function useScanSubscription(
    baseWebSocket: any,
    scanId: string | null,
    onUpdate: (data: any) => void
) {
    return useSubscription(
        baseWebSocket,
        scanId,
        'subscribe_scan',
        'unsubscribe_scan',
        (message) => message.type === 'scan_update' && message.payload?.scanId === scanId,
        onUpdate
    );
}

/**
 * Specialized hook for channel subscriptions (dashboard, etc.)
 */
export function useChannelSubscription(
    baseWebSocket: any,
    channel: string | null,
    onUpdate: (data: any) => void
) {
    return useSubscription(
        baseWebSocket,
        channel,
        'subscribe',
        'unsubscribe',
        (message) => {
            const channelTypes = ['stats', 'analytics', 'activity', 'metrics', 'all'];
            return channelTypes.includes(message.type);
        },
        onUpdate,
        { payload: { channel } }
    );
}
