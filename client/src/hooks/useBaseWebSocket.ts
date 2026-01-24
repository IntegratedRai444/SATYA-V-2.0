import { useEffect, useRef, useState, useCallback } from 'react';
import logger from '../lib/logger';

export interface WebSocketMessage {
    type: string;
    data?: unknown;
    payload?: unknown;
    error?: string;
    timestamp?: string;
}

export interface BaseWebSocketOptions {
    autoConnect?: boolean;
    reconnectAttempts?: number;
    reconnectInterval?: number;
    url?: string;
    onMessage?: (message: WebSocketMessage) => void;
    onError?: (error: Error) => void;
    onConnected?: () => void;
    onDisconnected?: () => void;
}

/**
 * Base WebSocket hook with core connection logic
 * Provides reconnection, subscription management, and message handling
 */
export function useBaseWebSocket(options: BaseWebSocketOptions = {}) {
    const {
        autoConnect = true,
        reconnectAttempts = 5,
        reconnectInterval = 3000,
        url,
        onMessage,
        onError,
        onConnected,
        onDisconnected,
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [connectionError, setConnectionError] = useState<string | null>(null);
    const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected' | 'error'>('disconnected');

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectCountRef = useRef(0);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set());
    const isMountedRef = useRef(true);

    // Get WebSocket URL
    const getWebSocketUrl = useCallback(() => {
        if (url) return url;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Use backend server instead of frontend host
        const wsHost = import.meta.env.VITE_WS_URL?.replace(/^ws:\/\//, '')?.replace(/^wss:\/\//, '') || 'localhost:5001';
        return `${protocol}//${wsHost}/api/v2/dashboard/ws`;
    }, [url]);

    // Handle incoming messages
    const handleMessage = useCallback((event: MessageEvent) => {
        try {
            const message: WebSocketMessage = JSON.parse(event.data);

            // Call registered handlers
            messageHandlersRef.current.forEach(handler => {
                try {
                    handler(message);
                } catch (error) {
                    logger.error('Error in message handler', error as Error);
                }
            });

            // Call option callback
            onMessage?.(message);
        } catch (error) {
            logger.error('Failed to parse WebSocket message', error as Error);
        }
    }, [onMessage]);

    // Handle errors
    const handleError = useCallback((error: Error | Event) => {
        const errorObj = error instanceof Error ? error : new Error('WebSocket connection failed');
        logger.error('WebSocket error', errorObj);
        setConnectionError(errorObj.message);
        setConnectionStatus('error');
        onError?.(errorObj);
    }, [onError]);

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        if (!isMountedRef.current) {
            return;
        }

        try {
            setConnectionStatus('connecting');
            const wsUrl = getWebSocketUrl();
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                if (!isMountedRef.current) return;

                logger.info('WebSocket connected');
                setIsConnected(true);
                setConnectionError(null);
                setConnectionStatus('connected');
                reconnectCountRef.current = 0;
                onConnected?.();
            };

            ws.onmessage = handleMessage;

            ws.onclose = (event) => {
                if (!isMountedRef.current) return;

                logger.info('WebSocket disconnected', { code: event.code, reason: event.reason });
                setIsConnected(false);
                setConnectionStatus('disconnected');
                wsRef.current = null;
                onDisconnected?.();

                // Attempt to reconnect if not a normal closure
                if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
                    scheduleReconnect();
                }
            };

            ws.onerror = (event) => {
                handleError(event);
            };

            wsRef.current = ws;
        } catch (error) {
            handleError(error as Error);
        }
    }, [getWebSocketUrl, handleMessage, handleError, reconnectAttempts, onConnected, onDisconnected]);

    // Schedule reconnection with exponential backoff
    const scheduleReconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        if (!isMountedRef.current) {
            return;
        }

        reconnectCountRef.current++;
        const delay = Math.min(
            reconnectInterval * Math.pow(2, reconnectCountRef.current - 1),
            30000 // Cap at 30 seconds
        );

        logger.info(`Reconnecting in ${delay}ms (attempt ${reconnectCountRef.current}/${reconnectAttempts})`);

        reconnectTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current) {
                connect();
            }
        }, delay);
    }, [connect, reconnectInterval, reconnectAttempts]);

    // Disconnect from WebSocket
    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close(1000, 'Manual disconnect');
            wsRef.current = null;
        }

        setIsConnected(false);
        setConnectionStatus('disconnected');
        reconnectCountRef.current = 0;
    }, []);

    // Send message through WebSocket
    const sendMessage = useCallback((message: unknown) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
            return true;
        } else {
            logger.warn('WebSocket is not connected');
            return false;
        }
    }, []);

    // Subscribe to messages
    const subscribe = useCallback((handler: (message: WebSocketMessage) => void) => {
        messageHandlersRef.current.add(handler);

        return () => {
            messageHandlersRef.current.delete(handler);
        };
    }, []);

    // Auto-connect on mount if enabled
    useEffect(() => {
        if (autoConnect) {
            connect();
        }

        return () => {
            isMountedRef.current = false;
            disconnect();
        };
    }, [autoConnect, connect, disconnect]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
        };
    }, []);

    return {
        isConnected,
        connectionError,
        connectionStatus,
        connect,
        disconnect,
        sendMessage,
        subscribe,
        reconnect: connect,
    };
}
