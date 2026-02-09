// Core API client - Consolidated single client implementation
export { api as default, apiClient } from './client';
export type { ApiResponse, DashboardStats, ExtendedDashboardStats } from './client';

// Services
export { default as analysisService, type AnalysisResult, type AnalysisOptions } from './services/analysisService';
export { default as dashboardService } from './services/dashboardService';

// Base service for extending
export { BaseService } from './services/baseService';

// Types
export type { RequestOptions } from './services/baseService';
