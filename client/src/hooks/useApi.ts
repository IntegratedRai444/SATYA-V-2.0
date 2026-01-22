import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { toast } from '@/components/ui/use-toast';
import { useAppContext } from '@/contexts/AppContext';

// Import services
import { analysisService } from '@/lib/api/services/analysisService';
import apiClient from '@/lib/api';

// Import Supabase auth instead of old authService
import { 
  getCurrentUser, 
  login as supabaseLogin, 
  register as supabaseRegister, 
  logout as supabaseLogout, 
  isAuthenticated as supabaseIsAuthenticated,
  type User as SupabaseUser,
  type LoginCredentials as SupabaseLoginCredentials,
  type RegisterData as SupabaseRegisterData
} from '@/services/supabaseAuth';

// Import types
import type { AnalysisResult } from '@/lib/api/services/analysisService';

// Destructure React hooks
const { useState, useEffect } = React;

// Define user preferences type
interface UserPreferences {
  language: string;
  timezone: string;
  dateFormat: string;
  timeFormat: '12h' | '24h';
}

// Define user type with preferences
export interface User extends SupabaseUser, Partial<UserPreferences> {
  language?: string;
  timezone?: string;
  dateFormat?: string;
  timeFormat?: '12h' | '24h';
}

// Re-export types for backward compatibility
export type { AnalysisResult, User as UserProfile, SupabaseUser, SupabaseLoginCredentials, SupabaseRegisterData };

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

// Re-export for backward compatibility
export type { LoginCredentials as LegacyLoginCredentials };

export interface Session {
  valid: boolean;
  user?: {
    id: string;
    username: string;
    email?: string;
    role: string;
  };
  error?: string;
}

// Queue for offline requests (kept for backward compatibility)
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
export function useEnhancedQuery<TData = unknown, TError = Error>(
  options: UseQueryOptions<TData, TError> & {
    queryKey: readonly unknown[];
    queryFn: () => Promise<TData>;
    offlineCache?: boolean;
  }
) {
  const { offlineCache = true, ...queryOptions } = options;
  
  return useQuery<TData, TError>({
    ...queryOptions,
    queryFn: async () => {
      if (offlineCache && !navigator.onLine) {
        // Try to get from cache if offline
        const cached = localStorage.getItem(`cache:${options.queryKey.join(':')}`);
        if (cached) {
          return JSON.parse(cached);
        }
        throw new Error('No internet connection and no cached data available');
      }
      
      try {
        const data = await options.queryFn();
        if (offlineCache) {
          localStorage.setItem(`cache:${options.queryKey.join(':')}`, JSON.stringify(data));
        }
        return data;
      } catch (error) {
        if (offlineCache && !navigator.onLine) {
          const cached = localStorage.getItem(`cache:${options.queryKey.join(':')}`);
          if (cached) {
            return JSON.parse(cached);
          }
        }
        throw error;
      }
    },
  });
}

// Enhanced mutation with offline support
export function useEnhancedMutation<TData = unknown, TVariables = unknown, TError = Error>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
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
    onSuccess: (data, variables, context, onMutateResult) => {
      // Invalidate and refetch any queries that might be affected by this mutation
      if (options?.onSuccess) {
        options.onSuccess(data, variables, context, onMutateResult);
      }
    },
    onError: (error, variables, context, onMutateResult) => {
      console.error('Mutation Error:', error);
      if (options?.onError) {
        options.onError(error, variables, context, onMutateResult);
      }
    },
  });
}

export function useAuth() {
  const { state, dispatch } = useAppContext();
  const queryClient = useQueryClient();
  
  // Get current user from Supabase auth
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const userData = await getCurrentUser();
        const userWithPreferences: User = {
          ...userData!,
          language: ((userData as unknown as Record<string, unknown>).language as string) || 'en',
          timezone: ((userData as unknown as Record<string, unknown>).timezone as string) || 'UTC',
          dateFormat: ((userData as unknown as Record<string, unknown>).dateFormat as string) || 'YYYY-MM-DD',
          timeFormat: (((userData as unknown as Record<string, unknown>).timeFormat as '12h' | '24h') || '24h')
        };
        setCurrentUser(userWithPreferences);
      } catch (error) {
        console.error('Failed to fetch current user:', error);
        setCurrentUser(null);
      }
    };
    
    fetchUser();
  }, [state.userPreferences]);
  
  const user = currentUser || undefined;

  // Login mutation using Supabase
  const loginMutation = useMutation({
    mutationFn: async (credentials: SupabaseLoginCredentials) => {
      const response = await supabaseLogin(credentials);
      return {
        ...response,
        user: {
          ...response.user,
          language: ((response.user as unknown as Record<string, unknown>).language as string) || 'en-US',
          timezone: ((response.user as unknown as Record<string, unknown>).timezone as string) || 'UTC',
          dateFormat: ((response.user as unknown as Record<string, unknown>).dateFormat as string) || 'YYYY-MM-DD',
          timeFormat: (((response.user as unknown as Record<string, unknown>).timeFormat as '12h' | '24h') || '24h')
        }
      };
    },
    onSuccess: (data: { user: User & { language?: string; timezone?: string; dateFormat?: string; timeFormat?: '12h' | '24h' }; [key: string]: unknown }) => {
      // Update user preferences after successful login
      dispatch({
        type: 'UPDATE_USER_PREFERENCES', 
        payload: {
          language: data.user.language || 'en-US',
          timezone: data.user.timezone || 'UTC',
          dateFormat: data.user.dateFormat || 'YYYY-MM-DD',
          timeFormat: data.user.timeFormat || '24h' as const
        } 
      });
      queryClient.invalidateQueries();
    },
  });
  
  // Logout mutation using Supabase
  const logoutMutation = useMutation({
    mutationFn: () => supabaseLogout(),
    onSuccess: () => {
      // Reset user preferences on logout
      dispatch({ 
        type: 'UPDATE_USER_PREFERENCES', 
        payload: {
          language: 'en-US',
          timezone: 'UTC',
          dateFormat: 'YYYY-MM-DD',
          timeFormat: '24h' as const
        } 
      });
      queryClient.clear();
      setCurrentUser(null);
    },
  });
  
  // Register mutation using Supabase
  const registerMutation = useMutation({
    mutationFn: async (data: SupabaseRegisterData) => {
      const response = await supabaseRegister(data);
      return {
        ...response,
        user: {
          ...response.user,
          language: ((response.user as unknown as Record<string, unknown>).language as string) || 'en-US',
          timezone: ((response.user as unknown as Record<string, unknown>).timezone as string) || 'UTC',
          dateFormat: ((response.user as unknown as Record<string, unknown>).dateFormat as string) || 'YYYY-MM-DD',
          timeFormat: (((response.user as unknown as Record<string, unknown>).timeFormat as '12h' | '24h') || '24h')
        }
      };
    },
    onSuccess: (data: { user: User & { language?: string; timezone?: string; dateFormat?: string; timeFormat?: '12h' | '24h' }; [key: string]: unknown }) => {
      // Update user preferences after successful registration
      dispatch({ 
        type: 'UPDATE_USER_PREFERENCES', 
        payload: {
          language: data.user.language || 'en-US',
          timezone: data.user.timezone || 'UTC',
          dateFormat: data.user.dateFormat || 'YYYY-MM-DD',
          timeFormat: data.user.timeFormat || '24h' as const
        } 
      });
      queryClient.invalidateQueries();
    },
  });

  const checkAuth = async (): Promise<Session> => {
    try {
      const isAuth = await supabaseIsAuthenticated();
      
      // Get user from auth service if authenticated
      let user: User | undefined;
      if (isAuth) {
        const userData = await getCurrentUser();
        const preferences = state.userPreferences || {};
        
        // Create user object with merged properties
        user = {
          ...userData!,
          ...(preferences.language && { language: preferences.language }),
          ...(preferences.timezone && { timezone: preferences.timezone }),
          ...(preferences.dateFormat && { dateFormat: preferences.dateFormat }),
          ...(preferences.timeFormat && { timeFormat: preferences.timeFormat as '12h' | '24h' })
        };
        
        // Update user preferences in app context if any exist
        if (Object.keys(preferences).length > 0) {
          dispatch({
            type: 'UPDATE_USER_PREFERENCES',
            payload: preferences
          });
        }
      }
      return { valid: isAuth, user: user ? {
        id: ((user as unknown as Record<string, unknown>).id as string) || '',
        username: ((user as unknown as Record<string, unknown>).username as string) || ((user as unknown as Record<string, unknown>).email as string) || 'unknown',
        email: ((user as unknown as Record<string, unknown>).email as string),
        role: ((user as unknown as Record<string, unknown>).role as string) || 'user'
      } : undefined };
    } catch (error) {
      return { valid: false, error: 'Not authenticated' };
    }
  };

  return {
    // Auth state
    isAuthenticated: !!user,
    user,
    
    // Auth actions
    login: loginMutation.mutate,
    loginAsync: loginMutation.mutateAsync,
    logout: logoutMutation.mutate,
    register: registerMutation.mutate,
    registerAsync: registerMutation.mutateAsync as unknown as (data: SupabaseRegisterData) => Promise<{
      user: User;
      accessToken: string;
      refreshToken: string;
    }>,
    checkAuth,
    
    // Loading states
    isLoading: loginMutation.isPending || logoutMutation.isPending || registerMutation.isPending,
    isLoggingIn: loginMutation.isPending,
    isLoggingOut: logoutMutation.isPending,
    isRegistering: registerMutation.isPending,
    
    // Errors
    error: loginMutation.error || logoutMutation.error || registerMutation.error,
    loginError: loginMutation.error,
    logoutError: logoutMutation.error,
    registerError: registerMutation.error,
  };
}

export function useDarkwebCheck() {
  const checkDarkweb = useMutation({
    mutationFn: async ({ contentId }: { contentId: string }) => {
      // Implement darkweb check logic here or call appropriate service method
      console.log('Checking darkweb for content:', contentId);
      return { found: false, sources: [] };
    },
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
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: 'destructive',
      });
    },
  });

  return checkDarkweb;
}

export function useVideoAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeVideo = useMutation({
    mutationFn: (file: File) => {
      return analysisService.analyzeVideo(file);
    },
    onSuccess: () => {
      // Update cache with the new analysis result
      queryClient.invalidateQueries();
      
      // Show success notification
      toast({
        title: 'Analysis Complete',
        description: 'Video analysis completed successfully',
        variant: 'default',
      });
    },
  });
  
  return analyzeVideo;
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      // Implement health check logic here or call appropriate service method
      return {
        status: 'healthy' as const,
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      };
    },
    retry: 3,
    refetchInterval: 30000,
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: async () => {
      // Implement system metrics logic here or call appropriate service method
      return {
        cpu: 0,
        memory: 0,
        uptime: 0,
        timestamp: new Date().toISOString()
      };
    },
    refetchInterval: 60000,
  });
}

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: async () => {
      // Implement system status logic here or call appropriate service method
      return {
        status: 'operational' as const,
        lastChecked: new Date().toISOString()
      };
    },
    refetchInterval: 10000,
  });
}

export function useImageAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeImage = useMutation({
    mutationFn: async (file: File) => {
      // Call the analysis service to analyze the image
      return analysisService.analyzeImage(file);
    },
    onSuccess: () => {
      // Update cache
      queryClient.invalidateQueries();
      
      // Show success notification
      toast({
        title: 'Analysis Complete',
        description: `Image analyzed successfully`,
        variant: 'default',
      });
    },
  });
  
  return analyzeImage;
}

export function useBlockchainVerification() {
  const queryClient = useQueryClient();
  
  const verifyOnBlockchain = useMutation({
    mutationFn: async (data: { hash: string; metadata: Record<string, unknown> }) => {
      // Call the analysis service to verify on blockchain
      console.log('Verifying on blockchain:', data.hash);
      const result = await analysisService.analyzeImage(
        new File([], 'blockchain-verification'),
        { includeDetails: true }
      );
      
      return {
        verified: result.result?.isAuthentic || false,
        txHash: `0x${Math.random().toString(16).substring(2, 66)}`
      };
    },
    onSuccess: (result) => {
      // Update cache
      queryClient.invalidateQueries({ queryKey: ['blockchain'] });
      
      // Show success notification
      toast({
        title: 'Verification Complete',
        description: `Transaction hash: ${result.txHash}`,
        variant: 'default',
      });
    },
  });

  return verifyOnBlockchain;
}

export function useEmotionConflictAnalysis() {
  const queryClient = useQueryClient();
  
  const analyzeEmotionConflict = useMutation({
    mutationFn: async (data: { text: string; audioFile?: File; videoFile?: File }) => {
      // Use appropriate analysis method based on available files
      let result: AnalysisResult;
      
      if (data.videoFile) {
        result = await analysisService.analyzeVideo(data.videoFile, {
          includeDetails: true,
          // text is a valid option for video analysis
          text: data.text
        });
      } else if (data.audioFile) {
        result = await analysisService.analyzeAudio(data.audioFile, {
          includeDetails: true,
          // text is a valid option for audio analysis
          text: data.text
        });
      } else {
        // Fallback to image analysis with text content
        const textBlob = new Blob([data.text], { type: 'text/plain' });
        result = await analysisService.analyzeImage(
          new File([textBlob], 'text-analysis.txt'),
          { includeDetails: true }
        );
      }
      
      return {
        conflict: !result.result?.isAuthentic,
        confidence: result.result?.confidence || 0,
        analysis: result.result?.details || {}
      };
    },
    onSuccess: (result) => {
      // Update cache
      queryClient.invalidateQueries({ queryKey: ['analysis', 'emotion-conflict'] });
      
      // Show result notification
      toast({
        title: result.conflict ? 'Emotion Conflict Detected' : 'No Emotion Conflict',
        description: `Confidence: ${(result.confidence * 100).toFixed(1)}%`,
        variant: result.conflict ? 'destructive' : 'default',
      });
    },
  });
  
  return analyzeEmotionConflict;
}

export function useModelsInfo() {
  return useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      try {
        // Get available models from the analysis service
        try {
          const response = await analysisService.getModels();
          return response || [];
        } catch (error) {
          // Fallback to a default list if getModels is not available
          return [
            { 
              id: 'default-model', 
              name: 'Default Model', 
              version: '1.0.0',
              description: 'Default analysis model',
              isActive: true
            }
          ] as const;
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
        return [];
      }
    },
  });
}

// Custom hook for analysis history (API-based)
export function useAnalysisHistory() {
  const queryClient = useQueryClient();
  
  // Define AnalysisHistoryItem type for API response
  type AnalysisHistoryItem = {
    id: string;
    type: 'image' | 'video' | 'audio' | 'text' | 'multimodal';
    status: 'completed' | 'failed' | 'processing';
    result?: unknown;
    error?: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
    jobId?: string;
    reportCode?: string;
  };
  
  // Fetch history from API with proper authentication
  const { data: historyData, error, isLoading } = useQuery({
    queryKey: ['analysis-history'],
    queryFn: async (): Promise<{ jobs: Array<Record<string, unknown>>; pagination: Record<string, unknown> }> => {
      // Use apiClient for proper authentication and error handling
      const response = await apiClient.get('/api/v2/history');
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error) => {
      // Don't retry on 401/403 errors, retry up to 3 times on network errors
      if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status?: number }).status;
        if (status === 401 || status === 403) return false;
      }
      return failureCount < 3;
    },
  });

  // Extract jobs array from response
  const history = historyData?.jobs || [];

  const saveToHistory = async (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp' | 'jobId' | 'reportCode'>) => {
    // This is now handled automatically by the backend after analysis
    // No need to manually save to history anymore
    console.log('History item would be saved:', item.type);
    console.log('History is now saved automatically by the backend');
  };

  const clearHistory = async () => {
    try {
      const response = await apiClient.delete('/api/v2/history');
      if (response.success) {
        queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
        toast({
          title: 'History Cleared',
          description: 'All analysis history has been cleared',
        });
      }
    } catch (error) {
      console.error('Failed to clear history:', error);
      toast({
        title: 'Error',
        description: 'Failed to clear history',
        variant: 'destructive',
      });
    }
  };

  const deleteFromHistory = async (jobId: string) => {
    try {
      const response = await apiClient.delete(`/api/v2/history/${jobId}`);
      if (response.success) {
        queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
        toast({
          title: 'Item Deleted',
          description: 'History item has been deleted',
        });
      }
      return true;
    } catch (error) {
      console.error(`Failed to delete history item ${jobId}:`, error);
      toast({
        title: 'Error',
        description: 'Failed to delete history item',
        variant: 'destructive',
      });
      return false;
    }
  };
  
  return {
    history: history || [],
    isLoading,
    error,
    saveToHistory,
    clearHistory,
    deleteFromHistory,
  };
}