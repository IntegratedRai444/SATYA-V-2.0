import { useState, useCallback, useEffect, useRef } from 'react';

export interface UseFetchOptions<TData> {
    enabled?: boolean;
    refetchInterval?: number;
    onSuccess?: (data: TData) => void;
    onError?: (error: Error) => void;
    retry?: number;
    retryDelay?: number;
    staleTime?: number;
}

export interface UseFetchResult<TData> {
    data: TData | null;
    isLoading: boolean;
    error: Error | null;
    refetch: () => Promise<void>;
    isRefetching: boolean;
}

/**
 * Generic fetch hook to eliminate duplication across multiple hooks
 * Replaces the fetch-error-loading pattern in:
 * - useAnalytics
 * - useDashboardStats
 * - useSettings
 * - useUser
 * - and 4+ more hooks
 * 
 * @param fetcher - Async function that fetches the data
 * @param options - Configuration options
 * @returns Fetch state and refetch function
 */
export function useFetch<TData = any>(
    fetcher: () => Promise<TData>,
    options: UseFetchOptions<TData> = {}
): UseFetchResult<TData> {
    const {
        enabled = true,
        refetchInterval,
        onSuccess,
        onError,
        retry = 0,
        retryDelay = 1000,
        staleTime = 0
    } = options;

    const [data, setData] = useState<TData | null>(null);
    const [isLoading, setIsLoading] = useState(enabled);
    const [isRefetching, setIsRefetching] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const retryCountRef = useRef(0);
    const lastFetchTimeRef = useRef<number>(0);
    const refetchIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const isMountedRef = useRef(true);

    const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const fetchData = useCallback(async (isRefetch = false) => {
        // Check if data is still fresh
        const now = Date.now();
        if (staleTime > 0 && data && (now - lastFetchTimeRef.current) < staleTime && !isRefetch) {
            return;
        }

        try {
            if (isRefetch) {
                setIsRefetching(true);
            } else {
                setIsLoading(true);
            }
            setError(null);

            const result = await fetcher();

            if (!isMountedRef.current) return;

            setData(result);
            lastFetchTimeRef.current = now;
            retryCountRef.current = 0;

            onSuccess?.(result);
        } catch (err) {
            if (!isMountedRef.current) return;

            const errorObj = err instanceof Error ? err : new Error('Fetch failed');

            // Retry logic
            if (retryCountRef.current < retry) {
                retryCountRef.current++;
                // Clear any existing retry timeout
                if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);

                retryTimeoutRef.current = setTimeout(() => {
                    if (isMountedRef.current) {
                        fetchData(isRefetch);
                    }
                }, retryDelay * retryCountRef.current);
                return;
            }

            setError(errorObj);
            onError?.(errorObj);
        } finally {
            if (isMountedRef.current) {
                setIsLoading(false);
                setIsRefetching(false);
            }
        }
    }, [fetcher, onSuccess, onError, retry, retryDelay, staleTime, data]);

    const refetch = useCallback(async () => {
        await fetchData(true);
    }, [fetchData]);

    // Initial fetch
    useEffect(() => {
        if (enabled) {
            fetchData();
        }
    }, [enabled, fetchData]);

    // Refetch interval
    useEffect(() => {
        if (refetchInterval && enabled) {
            refetchIntervalRef.current = setInterval(() => {
                fetchData(true);
            }, refetchInterval);

            return () => {
                if (refetchIntervalRef.current) {
                    clearInterval(refetchIntervalRef.current);
                }
            };
        }
    }, [refetchInterval, enabled, fetchData]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            isMountedRef.current = false;
            if (refetchIntervalRef.current) {
                clearInterval(refetchIntervalRef.current);
            }
            if (retryTimeoutRef.current) {
                clearTimeout(retryTimeoutRef.current);
            }
        };
    }, []);

    return {
        data,
        isLoading,
        error,
        refetch,
        isRefetching
    };
}

/**
 * Specialized hook for fetching with automatic retries
 */
export function useFetchWithRetry<TData = any>(
    fetcher: () => Promise<TData>,
    maxRetries = 3,
    options: Omit<UseFetchOptions<TData>, 'retry'> = {}
): UseFetchResult<TData> {
    return useFetch(fetcher, { ...options, retry: maxRetries });
}

/**
 * Specialized hook for polling data at intervals
 */
export function usePoll<TData = any>(
    fetcher: () => Promise<TData>,
    interval: number,
    options: Omit<UseFetchOptions<TData>, 'refetchInterval'> = {}
): UseFetchResult<TData> {
    return useFetch(fetcher, { ...options, refetchInterval: interval });
}
