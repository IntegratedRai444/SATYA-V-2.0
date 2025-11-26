/**
 * Custom Hook Type Definitions
 * Type definitions for all custom React hooks
 */

import type { AnalysisResult, AnalysisOptions, HistoryItem, DashboardStats } from './api';
import type { WebSocketMessage, ConnectionState } from './websocket';

// ============================================================================
// useAnalysis Hook Types
// ============================================================================

export interface UseAnalysisOptions extends AnalysisOptions {
  onSuccess?: (result: AnalysisResult) => void;
  onError?: (error: Error) => void;
  onProgress?: (progress: number) => void;
}

export interface UseAnalysisReturn {
  analyze: (file: File, options?: UseAnalysisOptions) => Promise<AnalysisResult>;
  isAnalyzing: boolean;
  progress: number;
  result: AnalysisResult | null;
  error: Error | null;
  reset: () => void;
}

// ============================================================================
// useWebSocket Hook Types
// ============================================================================

export interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  isConnected: boolean;
  send: <T>(message: T) => void;
  subscribe: (handler: (message: WebSocketMessage) => void) => () => void;
  connect: () => Promise<void>;
  disconnect: () => void;
  lastMessage: WebSocketMessage | null;
  error: Error | null;
}

// ============================================================================
// useBatchProcessing Hook Types
// ============================================================================

export interface BatchFile {
  fileId: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: AnalysisResult;
  error?: string;
}

export interface UseBatchProcessingOptions {
  maxConcurrent?: number;
  onFileComplete?: (fileId: string, result: AnalysisResult) => void;
  onFileError?: (fileId: string, error: Error) => void;
  onBatchComplete?: (results: AnalysisResult[]) => void;
}

export interface UseBatchProcessingReturn {
  files: BatchFile[];
  addFiles: (files: File[]) => void;
  removeFile: (fileId: string) => void;
  startProcessing: () => Promise<void>;
  pauseProcessing: () => void;
  resumeProcessing: () => void;
  clearCompleted: () => void;
  isProcessing: boolean;
  progress: number;
  completedCount: number;
  errorCount: number;
}

// ============================================================================
// useDashboard Hook Types
// ============================================================================

export interface UseDashboardOptions {
  refreshInterval?: number;
  autoRefresh?: boolean;
}

export interface UseDashboardReturn {
  stats: DashboardStats | null;
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  lastUpdated: string | null;
}

// ============================================================================
// useAnalytics Hook Types
// ============================================================================

export interface AnalyticsPeriod {
  start: string;
  end: string;
  label: string;
}

export interface UseAnalyticsOptions {
  period?: 'day' | 'week' | 'month' | 'year';
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface UseAnalyticsReturn {
  data: unknown | null;
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  setPeriod: (period: 'day' | 'week' | 'month' | 'year') => void;
  currentPeriod: AnalyticsPeriod;
}

// ============================================================================
// useHistory Hook Types
// ============================================================================

export interface UseHistoryOptions {
  limit?: number;
  autoLoad?: boolean;
  filters?: {
    type?: 'image' | 'video' | 'audio';
    result?: 'authentic' | 'manipulated' | 'uncertain';
    dateFrom?: string;
    dateTo?: string;
  };
}

export interface UseHistoryReturn {
  history: HistoryItem[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  loadMore: () => Promise<void>;
  refresh: () => Promise<void>;
  applyFilters: (filters: UseHistoryOptions['filters']) => void;
}

// ============================================================================
// useNotifications Hook Types
// ============================================================================

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
  actionLabel?: string;
}

export interface UseNotificationsOptions {
  maxNotifications?: number;
  autoMarkAsRead?: boolean;
  persistToStorage?: boolean;
}

export interface UseNotificationsReturn {
  notifications: Notification[];
  unreadCount: number;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

// ============================================================================
// useLocalStorage Hook Types
// ============================================================================

export interface UseLocalStorageOptions<T> {
  serializer?: (value: T) => string;
  deserializer?: (value: string) => T;
  initializeWithValue?: boolean;
}

export type UseLocalStorageReturn<T> = [
  T,
  (value: T | ((prev: T) => T)) => void,
  () => void
];

// ============================================================================
// useDebounce Hook Types
// ============================================================================

export interface UseDebounceOptions {
  leading?: boolean;
  trailing?: boolean;
  maxWait?: number;
}

export type UseDebounceReturn<T> = T;

// ============================================================================
// useThrottle Hook Types
// ============================================================================

export interface UseThrottleOptions {
  leading?: boolean;
  trailing?: boolean;
}

export type UseThrottleReturn<T> = T;

// ============================================================================
// useMediaQuery Hook Types
// ============================================================================

export type UseMediaQueryReturn = boolean;

// ============================================================================
// useSettings Hook Types
// ============================================================================

export interface AppSettings {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  notifications: {
    enabled: boolean;
    sound: boolean;
    desktop: boolean;
  };
  privacy: {
    analytics: boolean;
    crashReports: boolean;
  };
  performance: {
    animations: boolean;
    autoplay: boolean;
  };
}

export interface UseSettingsReturn {
  settings: AppSettings;
  updateSettings: (updates: Partial<AppSettings>) => Promise<void>;
  resetSettings: () => Promise<void>;
  isLoading: boolean;
  error: Error | null;
}

// ============================================================================
// useAuth Hook Types
// ============================================================================

export interface UseAuthReturn {
  user: {
    id: number;
    username: string;
    email: string;
    role: string;
  } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  error: Error | null;
}

// ============================================================================
// useApi Hook Types
// ============================================================================

export interface UseApiOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  retry?: number;
  retryDelay?: number;
}

export interface UseApiReturn<T, P extends unknown[] = []> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  execute: (...params: P) => Promise<T>;
  reset: () => void;
}

// ============================================================================
// Utility Function Types
// ============================================================================

export type DebouncedFunction<T extends (...args: unknown[]) => unknown> = {
  (...args: Parameters<T>): void;
  cancel: () => void;
  flush: () => void;
};

export type ThrottledFunction<T extends (...args: unknown[]) => unknown> = {
  (...args: Parameters<T>): void;
  cancel: () => void;
};

export type MemoizedFunction<T extends (...args: unknown[]) => unknown> = {
  (...args: Parameters<T>): ReturnType<T>;
  cache: Map<string, ReturnType<T>>;
  clear: () => void;
};
