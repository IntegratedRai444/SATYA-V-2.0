export type Database = {
  public: {
    Tables: {
      // Removed users table - we use auth.users directly
      analysis_jobs: {
        Row: {
          id: string;
          user_id: string; // UUID
          status: 'pending' | 'processing' | 'completed' | 'failed';
          media_type: string;
          file_name: string;
          file_path: string;
          file_size?: number;
          file_hash?: string;
          progress: number;
          metadata?: Record<string, any>;
          error_message?: string;
          priority: number;
          retry_count: number;
          report_code?: string;
          started_at?: string;
          completed_at?: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          status?: 'pending' | 'processing' | 'completed' | 'failed';
          media_type: string;
          file_name: string;
          file_path: string;
          file_size?: number;
          file_hash?: string;
          progress?: number;
          metadata?: Record<string, any>;
          error_message?: string;
          priority?: number;
          retry_count?: number;
          report_code?: string;
          started_at?: string;
          completed_at?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          status?: 'pending' | 'processing' | 'completed' | 'failed';
          media_type?: string;
          file_name?: string;
          file_path?: string;
          file_size?: number;
          file_hash?: string;
          progress?: number;
          metadata?: Record<string, any>;
          error_message?: string;
          priority?: number;
          retry_count?: number;
          report_code?: string;
          started_at?: string;
          completed_at?: string;
          updated_at?: string;
        };
      };
      analysis_results: {
        Row: {
          id: string;
          job_id: string;
          model_name: string;
          confidence?: number;
          is_deepfake: boolean;
          analysis_data: Record<string, any>;
          created_at: string;
        };
        Insert: {
          id?: string;
          job_id: string;
          model_name?: string;
          confidence?: number;
          is_deepfake: boolean;
          analysis_data: Record<string, any>;
          created_at?: string;
        };
        Update: {
          id?: string;
          job_id?: string;
          model_name?: string;
          confidence?: number;
          is_deepfake?: boolean;
          analysis_data?: Record<string, any>;
          created_at?: string;
        };
      };
      notifications: {
        Row: {
          id: string;
          user_id: string; // UUID
          type: 'info' | 'success' | 'warning' | 'error';
          title: string;
          message: string;
          is_read: boolean;
          action_url?: string;
          created_at: string;
          read_at?: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          type?: 'info' | 'success' | 'warning' | 'error';
          title: string;
          message: string;
          is_read?: boolean;
          action_url?: string;
          created_at?: string;
          read_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          type?: 'info' | 'success' | 'warning' | 'error';
          title?: string;
          message?: string;
          is_read?: boolean;
          action_url?: string;
          read_at?: string;
        };
      };
      scans: {
        Row: {
          id: number;
          user_id: string; // UUID
          filename: string;
          type: string;
          result: string;
          confidence_score: number;
          detection_details?: Record<string, any>;
          metadata?: Record<string, any>;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: number;
          user_id: string;
          filename: string;
          type: string;
          result: string;
          confidence_score: number;
          detection_details?: Record<string, any>;
          metadata?: Record<string, any>;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: number;
          user_id?: string;
          filename?: string;
          type?: string;
          result?: string;
          confidence_score?: number;
          detection_details?: Record<string, any>;
          metadata?: Record<string, any>;
          updated_at?: string;
        };
      };
      user_preferences: {
        Row: {
          id: number;
          user_id: string; // UUID
          theme?: string;
          language?: string;
          confidence_threshold?: number;
          enable_notifications?: boolean;
          auto_analyze?: boolean;
          sensitivity_level?: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: number;
          user_id: string;
          theme?: string;
          language?: string;
          confidence_threshold?: number;
          enable_notifications?: boolean;
          auto_analyze?: boolean;
          sensitivity_level?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: number;
          user_id?: string;
          theme?: string;
          language?: string;
          confidence_threshold?: number;
          enable_notifications?: boolean;
          auto_analyze?: boolean;
          sensitivity_level?: string;
          updated_at?: string;
        };
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
  };
};