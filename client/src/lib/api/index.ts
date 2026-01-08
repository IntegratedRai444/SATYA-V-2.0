// Core API client
export { api as default, apiClient, authApiClient, cancelAllRequests } from './client';

// Services
export { default as authService, type AuthResponse, type LoginCredentials, type RegisterData } from './services/authService';
export { default as analysisService, type AnalysisResult, type AnalysisOptions } from './services/analysisService';
export { default as userService, type UserProfile, type UpdateProfileData, type ChangePasswordData } from './services/userService';

// Base service for extending
export { BaseService } from './services/baseService';

// Types
export type { RequestOptions } from './services/baseService';
