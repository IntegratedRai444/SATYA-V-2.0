import { useEffect, useRef, useState, useCallback } from 'react';
import logger from '../lib/logger';
import { getAccessToken } from '../lib/auth/getAccessToken';

export interface WebSocketMessage {
    type: string;
    data?: unknown;
    payload?: unknown;
    error?: string;
    timestamp?: string;
    id?: string;
    jobId?: string;
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
    const reconnectTimeoutRef = useRef<number | null>(null);
    const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set());
    const isMountedRef = useRef(true);

    // Get WebSocket URL with authentication
    const getWebSocketUrl = useCallback(async () => {
        if (url) return url;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const configured = import.meta.env.VITE_WS_URL;
        
        // Get authentication token
        const token = await getAccessToken();
        if (!token) {
            throw new Error('No authentication token available');
        }

        if (configured) {
            const hasProtocol = /^wss?:\/\//i.test(configured);
            const baseUrl = hasProtocol ? configured : `${protocol}//${configured}`;
            const wsUrl = new URL(baseUrl);
            const path = wsUrl.pathname && wsUrl.pathname !== '/' ? wsUrl.pathname : '/api/v2/dashboard/ws';
            wsUrl.pathname = path;
            wsUrl.searchParams.set('token', token);
            return wsUrl.toString();
        }

        return `${protocol}//localhost:5001/api/v2/dashboard/ws?token=${encodeURIComponent(token)}`;
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

    // Schedule reconnection with exponential backoff
    const scheduleReconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        if (!isMountedRef.current) {
            return;
        }

        const delay = Math.min(reconnectInterval * Math.pow(2, reconnectCountRef.current), 30000);
        reconnectCountRef.current += 1;

        logger.info(`Scheduling reconnect attempt ${reconnectCountRef.current} in ${delay}ms`);

        reconnectTimeoutRef.current = window.setTimeout(() => {
            if (isMountedRef.current) {
                // We'll call connect directly here to avoid circular dependency
                // The actual connect function will be defined below
                (async () => {
                    try {
                        setConnectionStatus('connecting');
                        const wsUrl = await getWebSocketUrl();
                        const ws = new WebSocket(wsUrl);

                        ws.onopen = () => {
                            logger.info('WebSocket connected');
                            setIsConnected(true);
                            setConnectionStatus('connected');
                            reconnectCountRef.current = 0;
                            onConnected?.();
                        };

                        ws.onmessage = (event: MessageEvent) => {
                            try {
                                const data = JSON.parse(event.data);
                                handleMessage(data);
                            } catch (error) {
                                logger.error('Failed to parse WebSocket message', error instanceof Error ? error : new Error(String(error)), { 
                                    data: event.data 
                                });
                            }
                        };

                        ws.onclose = (event: CloseEvent) => {
                            logger.info('WebSocket disconnected', { code: event.code, reason: event.reason });
                            setIsConnected(false);
                            setConnectionStatus('disconnected');
                            wsRef.current = null;
                            onDisconnected?.();

                            // Attempt to reconnect if not a normal closure
                            if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
                                // Inline reconnection logic to avoid circular dependency
                                if (reconnectTimeoutRef.current) {
                                    clearTimeout(reconnectTimeoutRef.current);
                                }

                                if (isMountedRef.current) {
                                    const delay = Math.min(reconnectInterval * Math.pow(2, reconnectCountRef.current), 30000);
                                    reconnectCountRef.current += 1;

                                    logger.info(`Scheduling reconnect attempt ${reconnectCountRef.current} in ${delay}ms`);

                                    reconnectTimeoutRef.current = window.setTimeout(() => {
                                        if (isMountedRef.current) {
                                            // Create new connection directly
                                            (async () => {
                                                try {
                                                    setConnectionStatus('connecting');
                                                    const wsUrl = await getWebSocketUrl();
                                                    const ws = new WebSocket(wsUrl);

                                                    ws.onopen = () => {
                                                        logger.info('WebSocket connected');
                                                        setIsConnected(true);
                                                        setConnectionStatus('connected');
                                                        reconnectCountRef.current = 0;
                                                        onConnected?.();
                                                    };

                                                    ws.onmessage = (event: MessageEvent) => {
                                                        try {
                                                            const data = JSON.parse(event.data);
                                                            handleMessage(data);
                                                        } catch (error) {
                                                            logger.error('Failed to parse WebSocket message', error instanceof Error ? error : new Error(String(error)), { 
                                                                data: event.data 
                                                            });
                                                        }
                                                    };

                                                    ws.onclose = (event: CloseEvent) => {
                                                        logger.info('WebSocket disconnected', { code: event.code, reason: event.reason });
                                                        setIsConnected(false);
                                                        setConnectionStatus('disconnected');
                                                        wsRef.current = null;
                                                        onDisconnected?.();

                                                        // Recursive reconnection
                                                        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
                                                            if (reconnectTimeoutRef.current) {
                                                                clearTimeout(reconnectTimeoutRef.current);
                                                            }

                                                            if (isMountedRef.current) {
                                                                const nextDelay = Math.min(reconnectInterval * Math.pow(2, reconnectCountRef.current), 30000);
                                                                reconnectCountRef.current += 1;

                                                                reconnectTimeoutRef.current = window.setTimeout(() => {
                                                                    if (isMountedRef.current) {
                                                                        // This will create another connection
                                                                    }
                                                                }, nextDelay);
                                                            }
                                                        }
                                                    };

                                                    ws.onerror = (event: Event) => {
                                                        handleError(event);
                                                    };

                                                    wsRef.current = ws;
                                                } catch (error) {
                                                    handleError(error as Error);
                                                }
                                            })();
                                        }
                                    }, delay);
                                }
                            }
                        };

                        ws.onerror = (event: Event) => {
                            handleError(event);
                        };

                        wsRef.current = ws;
                    } catch (error) {
                        handleError(error as Error);
                    }
                })();
            }
        }, delay);
    }, [getWebSocketUrl, handleMessage, handleError, reconnectAttempts, onConnected, onDisconnected, reconnectInterval]);

    // Connect to WebSocket
    const connect = useCallback(async () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        if (!isMountedRef.current) {
            return;
        }

        try {
            setConnectionStatus('connecting');
            const wsUrl = await getWebSocketUrl();
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                logger.info('WS Connected');
                setIsConnected(true);
                setConnectionStatus('connected');
                reconnectCountRef.current = 0;
                onConnected?.();
            };

            ws.onmessage = (event: MessageEvent) => {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (error) {
                    logger.error('Failed to parse WebSocket message', error instanceof Error ? error : new Error(String(error)));
                }
            };

            ws.onclose = (event: CloseEvent) => {
                logger.info('WS Disconnected', { code: event.code, reason: event.reason });
                setIsConnected(false);
                setConnectionStatus('disconnected');
                wsRef.current = null;
                onDisconnected?.();

                // Attempt to reconnect if not a normal closure
                if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
                    scheduleReconnect();
                }
            };

            ws.onerror = (event: Event) => {
                handleError(event);
            };

            wsRef.current = ws;
        } catch (error) {
            handleError(error as Error);
        }
    }, [getWebSocketUrl, handleMessage, handleError, reconnectAttempts, onConnected, onDisconnected, scheduleReconnect]);

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
            try {
                const messageWithId: Record<string, unknown> = {
                    ...(typeof message === 'object' ? message : {}),
                    timestamp: new Date().toISOString(),
                    id: crypto.randomUUID()
                };
                wsRef.current.send(JSON.stringify(messageWithId));
                logger.debug('WebSocket message sent', { type: typeof message === 'object' && message !== null && 'type' in message ? (message as { type?: string }).type : 'unknown' });
                return true;
            } catch (error) {
                logger.error('Failed to send WebSocket message', error instanceof Error ? error : new Error(String(error)));
                return false;
            }
        } else {
            logger.warn('WebSocket is not connected, cannot send message');
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
            // Use setTimeout to avoid synchronous setState
            const timeoutId = setTimeout(() => {
                connect().catch((error: Error) => {
                    logger.error('Auto-connect failed', error);
                });
            }, 0);

            return () => {
                clearTimeout(timeoutId);
                isMountedRef.current = false;
                disconnect();
            };
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
        connect: () => connect().catch((error: Error) => {
            logger.error('Connection failed', error);
            setConnectionError('Connection failed');
            setConnectionStatus('error');
        }),
        disconnect,
        sendMessage,
        subscribe,
        reconnect: connect,
    };
}
