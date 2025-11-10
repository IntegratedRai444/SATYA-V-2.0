import { useMutation, useQuery, useQueryClient, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import apiClient from '../lib/api';
import type { AnalysisResult } from '../lib/api';
import { toast } from '../hooks/use-toast';
import { login, logout, checkAuth } from '../lib/auth';
import type { AuthResponse } from '../types';
import { useAppContext } from '../contexts/AppContext';

interface LoginCredentials {
  username: string;
  password: string;
}

interface Session {
  valid: boolean;
  user?: {
    id: number;
    username: string;
    email?: string;
    role: string;
  };
  error?: string;
}

// Auth hooks
// Queue for offline requests
const requestQueue: Array<() => Promise<void>> = [];
let isProcessingQueue = false;

// Process queued requests when back online
const processQueue = async () => {
  if (isProcessingQueue || requestQueue.length === 0) return;
  
  isProcessingQueue = true;
  try {
    while (requestQueue.length > 0) {
      const request = requestQueue.shift();
      if (request) await request();
    }
  } finally {
    isProcessingQueue = false;
  }
};

// Listen for online/offline events
if (typeof window !== 'undefined') {
  window.addEventListener('online', processQueue);
}

// Enhanced query with offline support
export function useEnhancedQuery<TData = any, TError = Error>(
  options: UseQueryOptions<TData, TError> & {
    queryKey: any[];
    queryFn: () => Promise<TData>;
    offlineCache?: boolean;
  }
) {
  const { state } = useAppContext();
  
  return useQuery<TData, TError>({
    ...options,
    enabled: options.enabled !== false && state.isOnline,
    cacheTime: 1000 * 60 * 5, // 5 minutes
    staleTime: 1000 * 60, // 1 minute
    retry: 1,
    onError: (error: any) => {
      console.error('Query Error:', error.message);
      if (options.onError) {
        options.onError(error);
      }
    },
  });
}

// Enhanced mutation with offline support
export function useEnhancedMutation<TData = any, TVariables = any, TError = Error>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  const queryClient = useQueryClient();
  const { state, dispatch } = useAppContext();
  
  return useMutation<TData, TError, TVariables>({
    mutationFn: async (variables) => {
      if (!state.isOnline) {
        // Queue the mutation for when we're back online
        return new Promise((resolve, reject) => {
          const request = async () => {
            try {
              const result = await mutationFn(variables);
              resolve(result);
            } catch (error) {
              reject(error);
            }
          };
          requestQueue.push(request);
          
          // Show offline notification
          dispatch({
            type: 'ADD_NOTIFICATION',
            payload: {
              id: `offline-${Date.now()}`,
              message: 'You are offline. Your request will be processed when you are back online.',
              type: 'info',
              duration: 5000,
            },
          });
          
          throw new Error('You are currently offline. Your request is queued and will be processed when you are back online.');
        });
      }
      return mutationFn(variables);
    },
    ...options,
    onSuccess: (data, variables, context) => {
      // Invalidate and refetch any queries that might be affected by this mutation
      queryClient.invalidateQueries();
      if (options?.onSuccess) {
        options.onSuccess(data, variables, context);
      }
    },
    onError: (error, variables, context) => {
      console.error('Mutation Error:', error);
      if (options?.onError) {
        options.onError(error, variables, context);
      }
    },
  });
}

export function useAuth() {
  const queryClient = useQueryClient();
  
  const { data: session, isLoading, error } = useEnhancedQuery<Session>({
    queryKey: ['auth'],
    queryFn: async () => {
      try {
        const response = await checkAuth();
        return {
          valid: response.success || false,
          user: response.user,
          error: response.error
        };
      } catch (err) {
        return {
          valid: false,
          error: err instanceof Error ? err.message : 'Authentication failed'
        };
      }
    },
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const loginMutation = useEnhancedMutation<AuthResponse, LoginCredentials, Error>(
    ({ username, password }) => login(username, password),
    {
      onSuccess: (data: AuthResponse) => {
        if (data.success) {
          queryClient.invalidateQueries({ queryKey: ['auth'] });
          toast({
            title: 'Login Successful',
            description: `Welcome back, ${data.user?.username || 'User'}!`,
          });
        } else {
          throw new Error(data.message || 'Login failed');
        }
      },
      onError: (error: Error) => {
        toast({
          title: 'Login Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  const logoutMutation = useEnhancedMutation<{ success: boolean }, void, Error>(
    () => logout(),
    {
      onSuccess: () => {
        queryClient.clear();
        toast({
          title: 'Logged Out',
          description: 'You have been successfully logged out.',
        });
      },
    }
  );

  return {
    isAuthenticated: session?.valid || false,
    user: session?.user,
    isLoading,
    error,
    login: loginMutation.mutate,
    logout: logoutMutation.mutate,
    isLoggingIn: loginMutation.isPending,
    isLoggingOut: logoutMutation.isPending,
  };
}

interface ModelInfo {
  id: string;
  name: string;
  version: string;
  description: string;
  isActive: boolean;
}

interface AnalysisHistoryItem {
  id: string;
  type: 'image' | 'video' | 'audio' | 'multimodal';
  result: any;
  timestamp: string;
  isOffline?: boolean;
}

type SystemStatus = {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  uptime: number;
  version: string;
};

export function useDarkwebCheck() {
  const queryClient = useQueryClient();
  
  const checkDarkweb = useEnhancedMutation<{ found: boolean; sources: string[] }, { contentId: string }, Error>(
    ({ contentId }) =>
      apiClient
        .post<{ found: boolean; sources: string[] }>('/check/darkweb', { contentId })
        .then((res) => res.data),
    {
      onSuccess: (data) => {
        toast({
          title: 'Darkweb Check Complete',
          description: data.found ? `Found ${data.sources.length} matches on darkweb` : 'No darkweb matches found',
          variant: data.found ? 'destructive' : 'default',
        });
      },
      onError: (error) => {
        toast({
          title: 'Darkweb Check Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  return checkDarkweb;
}

export function useVideoAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeVideo = useEnhancedMutation<AnalysisResult, FormData, Error>(
    (formData) =>
      apiClient.post<AnalysisResult>('/analyze/video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(res => res.data),
    {
      onSuccess: (data) => {
        toast({
          title: 'Video Analysis Complete',
          description: `Analysis completed with ${data.confidence.toFixed(1)}% confidence`,
        });
      },
      onError: (error) => {
        toast({
          title: 'Video Analysis Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  return analyzeVideo;
}

export function useHealth() {
  return useEnhancedQuery<HealthStatus>({
    queryKey: ['health'],
    queryFn: () => apiClient.get<HealthStatus>('/health').then(res => res.data),
    refetchInterval: 30000, // 30 seconds
    offlineCache: true,
  });
}

export function useMetrics() {
  return useEnhancedQuery<SystemMetrics>({
    queryKey: ['metrics'],
    queryFn: () => apiClient.get<SystemMetrics>('/metrics').then(res => res.data),
    refetchInterval: 60000, // 1 minute
    offlineCache: true,
  });
}

export function useStatus() {
  return useEnhancedQuery<SystemStatus>({
    queryKey: ['status'],
    queryFn: () => apiClient.get<SystemStatus>('/status').then(res => res.data),
    refetchInterval: 60000, // 1 minute
    offlineCache: true,
  });
}

export function useImageAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeImage = useEnhancedMutation<AnalysisResult, FormData, Error>(
    (formData) => 
      apiClient.post<AnalysisResult>('/analyze/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(res => res.data),
    {
      onSuccess: (data) => {
        toast({
          title: 'Image Analysis Complete',
          description: `Analysis completed with ${data.confidence.toFixed(1)}% confidence`,
        });
      },
      onError: (error) => {
        toast({
          title: 'Image Analysis Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  return analyzeImage;
}

export function useBlockchainVerification() {
  const queryClient = useQueryClient();
  
  const verifyOnChain = useEnhancedMutation<{ verified: boolean }, { contentId: string }, Error>(
    ({ contentId }) =>
      apiClient
        .post<{ verified: boolean }>('/verify/blockchain', { contentId })
        .then((res) => res.data),
    {
      onSuccess: (data) => {
        toast({
          title: 'Blockchain Verification Complete',
          description: data.verified ? 'Media verified on blockchain' : 'No blockchain record found',
        });
      },
      onError: (error) => {
        toast({
          title: 'Blockchain Verification Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  return verifyOnChain;
}

export function useEmotionConflictAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeEmotionConflict = useEnhancedMutation<{ hasConflict: boolean; confidence: number }, FormData, Error>(
    (formData) =>
      apiClient
        .post<{ hasConflict: boolean; confidence: number }>('/analyze/emotion-conflict', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        .then((res) => res.data),
    {
      onSuccess: (data) => {
        toast({
          title: 'Emotion Conflict Analysis Complete',
          description: `Face-voice emotion analysis completed with ${data.confidence.toFixed(1)}% confidence`,
        });
      },
      onError: (error) => {
        toast({
          title: 'Emotion Conflict Analysis Failed',
          description: error.message,
          variant: 'destructive',
        });
      },
    }
  );

  return analyzeEmotionConflict;
}

export function useModelsInfo() {
  return useEnhancedQuery<ModelInfo[]>({
    queryKey: ['models'],
    queryFn: () => apiClient.get<ModelInfo[]>('/models').then(res => res.data),
    staleTime: 5 * 60 * 1000, // 5 minutes
    offlineCache: true,
  });
}

// Custom hook for analysis history (local storage based)
export function useAnalysisHistory() {
  const { state } = useAppContext();
  const queryClient = useQueryClient();
  
  const getHistory = (): AnalysisHistoryItem[] => {
    if (typeof window === 'undefined') return [];
    const history = localStorage.getItem('analysisHistory');
    return history ? JSON.parse(history) : [];
  };

  const saveToHistory = (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp'>) => {
    if (!state.isOnline) {
      // Queue the save operation for when we're back online
      const offlineQueue = JSON.parse(localStorage.getItem('offlineQueue') || '[]');
      offlineQueue.push({
        type: 'saveToHistory',
        data: item,
        timestamp: new Date().toISOString(),
      });
      localStorage.setItem('offlineQueue', JSON.stringify(offlineQueue));
      
      // Return a temporary ID for UI purposes
      return {
        ...item,
        id: `offline-${Date.now()}`,
        timestamp: new Date().toISOString(),
        isOffline: true,
      };
    }
    
    const history = getHistory();
    const newItem = {
      ...item,
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
    };
    const newHistory = [newItem, ...history].slice(0, 50); // Keep only the last 50 items
    localStorage.setItem('analysisHistory', JSON.stringify(newHistory));
    return newItem;
  };

  const clearHistory = () => {
    if (!state.isOnline) {
      // Queue the clear operation for when we're back online
      const offlineQueue = JSON.parse(localStorage.getItem('offlineQueue') || '[]');
      offlineQueue.push({
        type: 'clearHistory',
        timestamp: new Date().toISOString(),
      });
      localStorage.setItem('offlineQueue', JSON.stringify(offlineQueue));
      return;
    }
    localStorage.removeItem('analysisHistory');
  };

  const deleteFromHistory = (id: string) => {
    if (!state.isOnline) {
      // Queue the delete operation for when we're back online
      const offlineQueue = JSON.parse(localStorage.getItem('offlineQueue') || '[]');
      offlineQueue.push({
        type: 'deleteFromHistory',
        data: { id },
        timestamp: new Date().toISOString(),
      });
      localStorage.setItem('offlineQueue', JSON.stringify(offlineQueue));
      
      // Return the current history without the deleted item for immediate UI update
      const history = getHistory();
      return history.filter((item) => item.id !== id);
    }
    
    const history = getHistory();
    const newHistory = history.filter((item) => item.id !== id);
    localStorage.setItem('analysisHistory', JSON.stringify(newHistory));
    return newHistory;
  };
  
  // Process offline queue when coming back online
  React.useEffect(() => {
    if (!state.isOnline) return;
    
    const processOfflineQueue = async () => {
      const offlineQueue = JSON.parse(localStorage.getItem('offlineQueue') || '[]');
      if (offlineQueue.length === 0) return;
      
      try {
        for (const item of offlineQueue) {
          switch (item.type) {
            case 'saveToHistory':
              await saveToHistory(item.data);
              break;
            case 'clearHistory':
              await clearHistory();
              break;
            case 'deleteFromHistory':
              await deleteFromHistory(item.data.id);
              break;
          }
        }
        // Clear the queue after processing
        localStorage.removeItem('offlineQueue');
      } catch (error) {
        console.error('Error processing offline queue:', error);
      }
    };
    
    processOfflineQueue();
  }, [state.isOnline, state]);

  return {
    history: getHistory(),
    saveToHistory,
    clearHistory,
    deleteFromHistory,
  };
}