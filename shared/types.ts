// Shared types for SATYA-V-2.0 application
// These types define the database schema structure

export type User = {
  id: number;
  username: string;
  password: string;
  email: string | null;
  full_name: string | null;
  api_key: string | null;
  role: string;
  failed_login_attempts: number;
  last_failed_login: string | null;
  is_locked: boolean;
  lockout_until: string | null;
  created_at: string;
  updated_at: string;
  last_login?: string | null;
};

export type Scan = {
  id: number;
  user_id: number;
  filename: string;
  type: string;
  result: string;
  confidence_score: number;
  detection_details: Record<string, any> | null;
  metadata: Record<string, any> | null;
  created_at: string;
  updated_at: string;
};

export type UserPreferences = {
  id: number;
  user_id: number;
  theme: string;
  language: string;
  confidence_threshold: number;
  enable_notifications: boolean;
  auto_analyze: boolean;
  sensitivity_level: string;
  created_at: string;
  updated_at: string;
};

export type AnalysisJob = {
  id: string;
  user_id: number;
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
  user_id: number;
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
