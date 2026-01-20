// Core API client
export { api as default, apiClient, authApiClient } from './client';
export type { ApiResponse, DashboardStats, ExtendedDashboardStats } from './client';

// Services
export { default as authService, type AuthResponse, type LoginCredentials, type RegisterData } from './services/authService';
export { default as analysisService, type AnalysisResult, type AnalysisOptions } from './services/analysisService';
export { default as dashboardService } from './services/dashboardService';

// Base service for extending
export { BaseService } from './services/baseService';

// Types
export type { RequestOptions } from './services/baseService';
