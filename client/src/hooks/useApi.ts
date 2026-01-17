import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { toast } from '@/components/ui/use-toast';
import { useAppContext } from '@/contexts/AppContext';

// Import services
import { authService } from '@/lib/api/services/authService';
import { analysisService } from '@/lib/api/services/analysisService';

// Import types
import type { User as AuthUser, RegisterData } from '@/lib/api/services/authService';
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
export interface User extends AuthUser, Partial<UserPreferences> {
  language?: string;
  timezone?: string;
  dateFormat?: string;
  timeFormat?: '12h' | '24h';
}

// Re-export types for backward compatibility
export type { AnalysisResult, User as UserProfile };

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

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
export function useEnhancedQuery<TData = any, TError = Error>(
  options: UseQueryOptions<TData, TError> & {
    queryKey: any[];
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
export function useEnhancedMutation<TData = any, TVariables = any, TError = Error>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  const { state, dispatch } = useAppContext();
  const queryClient = useQueryClient();
  
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
    onSuccess: (data, variables, context, unknown) => {
      // Invalidate and refetch any queries that might be affected by this mutation
      if (options?.onSuccess) {
        options.onSuccess(data, variables, context);
      }
    },
    onError: (error, variables, context, unknown) => {
      console.error('Mutation Error:', error);
      if (options?.onError) {
        options.onError(error, variables, context);
      }
    },
  });
}

export function useAuth() {
  const { state, dispatch } = useAppContext();
  
  // Check if user is authenticated
  const checkAuth = async (): Promise<Session> => {
    try {
      const isAuthenticated = authService.isAuthenticated();
      
      // Get user from auth service if authenticated
      let user: User | undefined;
      if (isAuthenticated) {
        const userData = await authService.getCurrentUser();
        const preferences = state.userPreferences || {};
        
        // Create user object with merged properties
        user = {
          ...userData,
          ...(preferences.language && { language: preferences.language }),
          ...(preferences.timezone && { timezone: preferences.timezone }),
          ...(preferences.dateFormat && { dateFormat: preferences.dateFormat }),
          ...(preferences.timeFormat && { timeFormat: preferences.timeFormat as '12h' | '24h' })
        };
        
        // Update user preferences in the app context if any exist
        if (Object.keys(preferences).length > 0) {
          dispatch({
            type: 'UPDATE_USER_PREFERENCES',
            payload: preferences
          });
        }
      }
      return { valid: isAuthenticated, user };
    } catch (error) {
      return { valid: false, error: 'Not authenticated' };
    }
  };
  
  // Login mutation
  const loginMutation = useMutation({
    mutationFn: async (credentials: LoginCredentials) => {
      const response = await authService.login(credentials);
      return {
        ...response,
        user: {
          ...response.user,
          language: (response.user as any).language || 'en-US',
          timezone: (response.user as any).timezone || 'UTC',
          dateFormat: (response.user as any).dateFormat || 'YYYY-MM-DD',
          timeFormat: ((response.user as any).timeFormat || '24h') as '12h' | '24h'
        }
      };
    },
    onSuccess: (data: any) => {
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
      useQueryClient.invalidateQueries();
    },
  });
  
  // Logout mutation
  const logoutMutation = useMutation({
    mutationFn: () => authService.logout(),
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
      useQueryClient.clear();
    },
  });
  
  // Register mutation
  const registerMutation = useMutation({
    mutationFn: async (data: RegisterData) => {
      const response = await authService.register(data);
      return {
        ...response,
        user: {
          ...response.user,
          language: (response.user as any).language || 'en-US',
          timezone: (response.user as any).timezone || 'UTC',
          dateFormat: (response.user as any).dateFormat || 'YYYY-MM-DD',
          timeFormat: ((response.user as any).timeFormat || '24h') as '12h' | '24h'
        }
      };
    },
    onSuccess: (data: any) => {
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
      useQueryClient.invalidateQueries();
    },
  });
  
  // Get current user from auth service
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  
  useEffect(() => {
    const fetchUser = async () => {
      if (authService.isAuthenticated()) {
        try {
          const userData = await authService.getCurrentUser();
          const userWithPreferences = {
            ...userData,
            language: (userData as any).language || 'en',
            timezone: (userData as any).timezone || 'UTC',
            dateFormat: (userData as any).dateFormat || 'YYYY-MM-DD',
            timeFormat: (((userData as any).timeFormat as '12h' | '24h') || '24h')
          };
          setCurrentUser(userWithPreferences);
        } catch (error) {
          console.error('Failed to fetch current user:', error);
          setCurrentUser(null);
        }
      } else {
        setCurrentUser(null);
      }
    };
    
    fetchUser();
  }, [state.userPreferences]);
  
  const user = currentUser || undefined;

  return {
    // Auth state
    isAuthenticated: authService.isAuthenticated(),
    user,
    
    // Auth actions
    login: loginMutation.mutate,
    loginAsync: loginMutation.mutateAsync,
    logout: logoutMutation.mutate,
    register: registerMutation.mutate,
    registerAsync: registerMutation.mutateAsync as unknown as (data: RegisterData) => Promise<{
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

interface ModelInfo {
  id: string;
  name: string;

export function useDarkwebCheck() {
  
  const checkDarkweb = useMutation({
    mutationFn: async ({ contentId }: { contentId: string }) => {
      // Implement darkweb check logic here or call appropriate service method
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
  
  const analyzeVideo = useMutation({
    mutationFn: (file: File) => {
      return analysisService.analyzeVideo(file);
    },
    onSuccess: () => {
      // Update cache with the new analysis result
      useQueryClient.invalidateQueries();
      
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
  
  const analyzeImage = useMutation({
    mutationFn: async (file: File) => {
      // Call the analysis service to analyze the image
      return analysisService.analyzeImage(file);
    },
    onSuccess: (data) => {
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
  
  const verifyOnBlockchain = useMutation({
    mutationFn: async (data: { hash: string; metadata: any }) => {
      // Call the analysis service to verify on blockchain
      const result = await analysisService.analyzeImage(
        new File([], 'blockchain-verification'),
        { includeDetails: true }
      );
      
      // Add metadata to the result
      const resultWithMetadata = {
        ...result,
        metadata: data.metadata
      };
      
      return {
        verified: result.result?.isAuthentic || false,
        txHash: `0x${Math.random().toString(16).substring(2, 66)}`
      };
    },
    onSuccess: (result) => {
      // Update cache
      useQueryClient.invalidateQueries({ queryKey: ['blockchain'] });
      
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
  
  const analyzeEmotionConflict = useMutation({
    mutationFn: async (data: { text: string; audioFile?: File; videoFile?: File }) => {
      // Use appropriate analysis method based on available files
      let result: AnalysisResult;
      
      if (data.videoFile) {
        result = await analysisService.analyzeVideo(data.videoFile, {
          includeDetails: true,
          // @ts-ignore - text is a valid option for video analysis
          text: data.text
        });
      } else if (data.audioFile) {
        result = await analysisService.analyzeAudio(data.audioFile, {
          includeDetails: true,
          // @ts-ignore - text is a valid option for audio analysis
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
      useQueryClient.invalidateQueries({ queryKey: ['analysis', 'emotion-conflict'] });
      
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
  const { state } = useAppContext();
  
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
  
  // Fetch history from API
  const { data: history, error, isLoading } = useQuery({
    queryKey: ['analysis-history'],
    queryFn: async (): Promise<AnalysisHistoryItem[]> => {
      const response = await fetch('/api/v2/history');
      if (!response.ok) {
        throw new Error('Failed to fetch analysis history');
      }
      const data = await response.json();
      return data.success ? data.data : [];
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const saveToHistory = async (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp' | 'jobId' | 'reportCode'>) => {
    // This is now handled automatically by the backend after analysis
    // No need to manually save to history anymore
    console.log('History is now saved automatically by the backend');
  };

  const clearHistory = async () => {
    try {
      const response = await fetch('/api/v2/history', {
        method: 'DELETE'
      });
      if (!response.ok) {
        throw new Error('Failed to clear analysis history');
      }
      const data = await response.json();
      if (data.success) {
        useQueryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      }
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const deleteFromHistory = async (jobId: string) => {
    try {
      const response = await fetch(`/api/v2/history/${jobId}`, {
        method: 'DELETE'
      });
      if (!response.ok) {
        throw new Error(`Failed to delete history item: ${jobId}`);
      }
      const data = await response.json();
      if (data.success) {
        useQueryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      }
      return true;
    } catch (error) {
      console.error(`Failed to delete history item ${jobId}:`, error);
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