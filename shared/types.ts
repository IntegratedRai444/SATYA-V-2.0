// Shared types for SATYA-V-2.0 application
// These types define the database schema structure

export type User = {
  id: string; // UUID
  username: string;
  email: string | null;
  full_name: string | null;
  avatar_url: string | null;
  role: 'user' | 'admin' | 'moderator';
  is_active: boolean;
  last_login: string | null;
  failed_login_attempts: number;
  last_failed_login: string | null;
  locked_until: string | null;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
};

export type Task = {
  id: string; // UUID
  user_id: string; // UUID
  type: 'analysis' | 'batch' | 'cleanup' | 'export';
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  file_name: string;
  file_size: number;
  file_type: string;
  file_path: string;
  report_code: string;
  result: any; // JSONB
  error: any; // JSONB
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
  metadata: Record<string, any>;
};

export type UserPreferences = {
  id: string; // UUID
  user_id: string; // UUID
  theme: 'light' | 'dark' | 'auto';
  language: string;
  confidence_threshold: number; // 0-100
  enable_notifications: boolean;
  auto_analyze: boolean;
  sensitivity_level: 'low' | 'medium' | 'high';
  chat_model: string | null;
  chat_enabled: boolean;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
};

// Legacy AnalysisJob type (deprecated - use Task instead)
export type AnalysisJob = {
  id: string;
  user_id: string; // UUID
  status: string;
  media_type: string;
  file_name: string;
  file_path: string;
  file_size: number | null;
  file_hash: string | null;
  progress: number;
  metadata: Record<string, any> | null;
  error_message: string | null;
  priority: number;
  retry_count: number;
  report_code: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
};

export type AnalysisResult = {
  id: string;
  job_id: string;
  user_id: string; // UUID
  result_type: string;
  confidence_score: number;
  details: Record<string, any> | null;
  metadata: Record<string, any> | null;
  created_at: string;
  updated_at: string;
};

// Supabase Auth user type
export type AuthUser = {
  id: string;
  email?: string;
  phone?: string;
  user_metadata?: Record<string, any>;
  app_metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
};

// API Response types
export type ApiResponse<T = any> = {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
};

// Auth types
export type LoginCredentials = {
  email: string;
  password: string;
};

export type RegisterData = {
  email: string;
  password: string;
  username: string;
};

export type AuthResponse = {
  user: AuthUser;
  session: {
    access_token: string;
    refresh_token: string;
    expires_at: number;
  };
};
