/**
 * React Query Performance Monitoring
 * Monitors and logs query performance metrics
 */

import { QueryClient } from '@tanstack/react-query';

export interface QueryPerformanceMetrics {
  queryKey: string;
  duration: number;
  status: 'success' | 'error' | 'loading' | 'pending';
  timestamp: number;
}

const performanceMetrics: QueryPerformanceMetrics[] = [];

/**
 * Monitor React Query performance
 */
export function monitorQueryPerformance(queryClient: QueryClient) {
  // Subscribe to query cache updates
  const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
    if (event?.type === 'updated' && event.query) {
      const query = event.query;
      const queryKey = JSON.stringify(query.queryKey);
      
      const startTime = (query.state.fetchMeta as any)?.startedAt || query.state.dataUpdatedAt;
      const duration = query.state.dataUpdatedAt ? query.state.dataUpdatedAt - startTime : 0;
      
      const metric: QueryPerformanceMetrics = {
        queryKey,
        duration,
        status: query.state.status,
        timestamp: Date.now(),
      };

      performanceMetrics.push(metric);

      // Keep only last 100 metrics
      if (performanceMetrics.length > 100) {
        performanceMetrics.shift();
      }

      // Log slow queries (> 2 seconds)
      if (metric.duration > 2000) {
        console.warn(`Slow query detected: ${queryKey} took ${metric.duration}ms`);
      }
    }
  });

  return unsubscribe;
}

/**
 * Get performance metrics
 */
export function getPerformanceMetrics(): QueryPerformanceMetrics[] {
  return [...performanceMetrics];
}

/**
 * Clear performance metrics
 */
export function clearPerformanceMetrics(): void {
  performanceMetrics.length = 0;
}
