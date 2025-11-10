/**
 * React Query configuration for caching and data management
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Create query client with optimized settings
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache for 5 minutes
      staleTime: 5 * 60 * 1000,
      // Keep in cache for 10 minutes
      cacheTime: 10 * 60 * 1000,
      // Retry failed requests
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.status >= 400 && error?.status < 500) {
          return false;
        }
        return failureCount < 3;
      },
      // Refetch on window focus for critical data
      refetchOnWindowFocus: false,
      // Background refetch
      refetchOnReconnect: true,
    },
    mutations: {
      // Retry mutations once
      retry: 1,
    },
  },
});

// Query keys for consistent caching
export const queryKeys = {
  // Analysis results
  analysisResult: (id: string) => ['analysis', 'result', id] as const,
  analysisHistory: (userId?: string) => ['analysis', 'history', userId] as const,
  analysisStats: () => ['analysis', 'stats'] as const,
  
  // User data
  userProfile: (userId: string) => ['user', 'profile', userId] as const,
  userSettings: (userId: string) => ['user', 'settings', userId] as const,
  
  // System data
  systemHealth: () => ['system', 'health'] as const,
  modelInfo: () => ['system', 'models'] as const,
  
  // Dashboard data
  dashboardStats: (timeRange?: string) => ['dashboard', 'stats', timeRange] as const,
  recentActivity: (limit?: number) => ['dashboard', 'activity', limit] as const,
} as const;

// Custom hooks for common queries
export const useAnalysisResult = (id: string) => {
  return useQuery({
    queryKey: queryKeys.analysisResult(id),
    queryFn: () => api.getAnalysisResult(id),
    enabled: !!id,
    staleTime: 10 * 60 * 1000, // Analysis results are stable
  });
};

export const useAnalysisHistory = (userId?: string) => {
  return useQuery({
    queryKey: queryKeys.analysisHistory(userId),
    queryFn: () => api.getAnalysisHistory(userId),
    staleTime: 2 * 60 * 1000, // Refresh every 2 minutes
  });
};

export const useDashboardStats = (timeRange = '7d') => {
  return useQuery({
    queryKey: queryKeys.dashboardStats(timeRange),
    queryFn: () => api.getDashboardStats(timeRange),
    staleTime: 5 * 60 * 1000,
  });
};

// Mutation hooks for data updates
export const useAnalysisMutation = () => {
  return useMutation({
    mutationFn: (data: AnalysisRequest) => api.submitAnalysis(data),
    onSuccess: (result, variables) => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.analysisHistory() });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboardStats() });
      
      // Cache the new result
      queryClient.setQueryData(
        queryKeys.analysisResult(result.id),
        result
      );
    },
  });
};

// Prefetch utilities
export const prefetchAnalysisHistory = async (userId?: string) => {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.analysisHistory(userId),
    queryFn: () => api.getAnalysisHistory(userId),
    staleTime: 2 * 60 * 1000,
  });
};

export const prefetchDashboardData = async () => {
  await Promise.all([
    queryClient.prefetchQuery({
      queryKey: queryKeys.dashboardStats(),
      queryFn: () => api.getDashboardStats(),
    }),
    queryClient.prefetchQuery({
      queryKey: queryKeys.recentActivity(10),
      queryFn: () => api.getRecentActivity(10),
    }),
  ]);
};

// Cache management utilities
export const clearAnalysisCache = () => {
  queryClient.removeQueries({ queryKey: ['analysis'] });
};

export const clearUserCache = (userId: string) => {
  queryClient.removeQueries({ queryKey: ['user', 'profile', userId] });
  queryClient.removeQueries({ queryKey: ['user', 'settings', userId] });
};

// Optimistic updates
export const updateAnalysisOptimistically = (id: string, updates: Partial<AnalysisResult>) => {
  queryClient.setQueryData(
    queryKeys.analysisResult(id),
    (old: AnalysisResult | undefined) => {
      if (!old) return undefined;
      return { ...old, ...updates };
    }
  );
};

// Background sync
export const syncAnalysisResults = async () => {
  const queries = queryClient.getQueryCache().findAll(['analysis', 'result']);
  
  for (const query of queries) {
    if (query.state.data) {
      try {
        await queryClient.refetchQueries({ queryKey: query.queryKey });
      } catch (error) {
        console.warn('Failed to sync analysis result:', query.queryKey, error);
      }
    }
  }
};

// Performance monitoring for queries
export const monitorQueryPerformance = () => {
  const cache = queryClient.getQueryCache();
  
  return {
    totalQueries: cache.getAll().length,
    activeQueries: cache.findAll({ type: 'active' }).length,
    staleQueries: cache.findAll({ stale: true }).length,
    cacheSize: cache.getAll().reduce((size, query) => {
      const data = query.state.data;
      return size + (data ? JSON.stringify(data).length : 0);
    }, 0),
  };
};

// Types
interface AnalysisRequest {
  type: 'image' | 'video' | 'audio' | 'multimodal';
  files: File[];
  options?: Record<string, any>;
}

interface AnalysisResult {
  id: string;
  type: string;
  authenticity: string;
  confidence: number;
  timestamp: string;
  // ... other fields
}

// Real API calls to backend
const api = {
  getAnalysisResult: async (id: string): Promise<AnalysisResult> => {
    const response = await fetch(`/api/analysis/${id}`);
    return response.json();
  },
  
  getAnalysisHistory: async (userId?: string): Promise<AnalysisResult[]> => {
    const url = userId ? `/api/analysis/history?userId=${userId}` : '/api/analysis/history';
    const response = await fetch(url);
    return response.json();
  },
  
  getDashboardStats: async (timeRange = '7d') => {
    const response = await fetch(`/api/dashboard/stats?range=${timeRange}`);
    return response.json();
  },
  
  getRecentActivity: async (limit = 10) => {
    const response = await fetch(`/api/dashboard/activity?limit=${limit}`);
    return response.json();
  },
  
  submitAnalysis: async (data: AnalysisRequest): Promise<AnalysisResult> => {
    const formData = new FormData();
    data.files.forEach((file, index) => {
      formData.append(`file${index}`, file);
    });
    formData.append('type', data.type);
    if (data.options) {
      formData.append('options', JSON.stringify(data.options));
    }
    
    const response = await fetch('/api/analysis/submit', {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  },
};