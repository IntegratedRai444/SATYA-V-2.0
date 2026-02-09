/**
 * Centralized URL Configuration
 * Eliminates hardcoded URLs and provides consistent URL management
 */

// Environment variables with proper typing
const getEnvVar = (name: string, required: boolean = true): string => {
  const value = import.meta.env[name];
  if (required && !value) {
    throw new Error(`Required environment variable ${name} is not set`);
  }
  return value || '';
};

// API Configuration
export const API_CONFIG = {
  // Base URLs from environment
  BASE_URL: getEnvVar('VITE_API_URL', false) || 'http://localhost:5001/api/v2',
  AUTH_URL: getEnvVar('VITE_AUTH_API_URL', false) || 'http://localhost:5001/api/v2',
  ANALYSIS_URL: getEnvVar('VITE_ANALYSIS_API_URL', false) || 'http://localhost:5001/api/v2',
  
  // WebSocket URL with automatic protocol detection
  WS_URL: getEnvVar('VITE_WS_URL', false) || (() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = getEnvVar('VITE_API_HOST', false) || 'localhost:5001';
    return `${protocol}//${host}/api/v2/dashboard/ws`;
  })(),
  
  // Supabase Configuration
  SUPABASE_URL: getEnvVar('VITE_SUPABASE_URL'),
  SUPABASE_ANON_KEY: getEnvVar('VITE_SUPABASE_ANON_KEY'),
  
  // Python ML Server (for reference only - frontend should not connect directly)
  // Note: All Python requests go through Node.js backend
  PYTHON_URL_REFERENCE: 'http://localhost:8000',
} as const;

// URL Builder Functions
export const buildApiUrl = (endpoint: string, baseUrl: string = API_CONFIG.BASE_URL): string => {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  const cleanBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  return `${cleanBaseUrl}/${cleanEndpoint}`;
};

export const buildWebSocketUrl = (token?: string): string => {
  const wsUrl = API_CONFIG.WS_URL;
  if (token) {
    const separator = wsUrl.includes('?') ? '&' : '?';
    return `${wsUrl}${separator}token=${encodeURIComponent(token)}`;
  }
  return wsUrl;
};

// Common Endpoints
export const API_ENDPOINTS = {
  // Authentication
  LOGIN: '/auth/login',
  REGISTER: '/auth/register',
  LOGOUT: '/auth/logout',
  REFRESH_TOKEN: '/auth/refresh-token',
  CSRF_TOKEN: '/auth/csrf-token',
  
  // Analysis
  ANALYZE_IMAGE: '/analysis/image',
  ANALYZE_VIDEO: '/analysis/video',
  ANALYZE_AUDIO: '/analysis/audio',
  ANALYZE_MULTIMODAL: '/analysis/multimodal',
  GET_ANALYSIS: '/analysis/',
  
  // Dashboard
  DASHBOARD_STATS: '/dashboard/stats',
  DASHBOARD_ACTIVITY: '/dashboard/recent-activity',
  DASHBOARD_ANALYTICS: '/dashboard/analytics',
  
  // File Upload
  UPLOAD_IMAGE: '/upload/image',
  UPLOAD_VIDEO: '/upload/video',
  UPLOAD_AUDIO: '/upload/audio',
  
  // History
  HISTORY: '/history',
  HISTORY_DELETE: '/history/',
} as const;

// Health Check URLs
export const HEALTH_URLS = {
  NODE_API: API_CONFIG.BASE_URL.replace('/api/v2', '') + '/health',
  PYTHON_ML: API_CONFIG.PYTHON_URL_REFERENCE + '/health',
  FRONTEND: window.location.origin,
} as const;

// Export default configuration
export default {
  ...API_CONFIG,
  buildApiUrl,
  buildWebSocketUrl,
  endpoints: API_ENDPOINTS,
  health: HEALTH_URLS,
};
