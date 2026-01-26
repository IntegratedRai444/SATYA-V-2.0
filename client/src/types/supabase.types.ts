export type Database = {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          username: string;
          email?: string;
          full_name?: string;
          avatar_url?: string;
          role: 'user' | 'admin' | 'moderator';
          is_active: boolean;
          last_login?: string;
          created_at: string;
          updated_at: string;
          deleted_at?: string;
        };
        Insert: {
          id?: string;
          username: string;
          email?: string;
          full_name?: string;
          avatar_url?: string;
          role?: 'user' | 'admin' | 'moderator';
          is_active?: boolean;
          last_login?: string;
          created_at?: string;
          updated_at?: string;
          deleted_at?: string;
        };
        Update: {
          id?: string;
          username?: string;
          email?: string;
          full_name?: string;
          avatar_url?: string;
          role?: 'user' | 'admin' | 'moderator';
          is_active?: boolean;
          last_login?: string;
          created_at?: string;
          updated_at?: string;
          deleted_at?: string | null;
        };
      };
      user_preferences: {
        Row: {
          id: string;
          user_id: string;
          theme: 'light' | 'dark' | 'auto';
          language: string;
          confidence_threshold: number;
          enable_notifications: boolean;
          auto_analyze: boolean;
          sensitivity_level: 'low' | 'medium' | 'high';
          chat_model?: string;
          chat_enabled: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          theme?: 'light' | 'dark' | 'auto';
          language?: string;
          confidence_threshold?: number;
          enable_notifications?: boolean;
          auto_analyze?: boolean;
          sensitivity_level?: 'low' | 'medium' | 'high';
          chat_model?: string;
          chat_enabled?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          theme?: 'light' | 'dark' | 'auto';
          language?: string;
          confidence_threshold?: number;
          enable_notifications?: boolean;
          auto_analyze?: boolean;
          sensitivity_level?: 'low' | 'medium' | 'high';
          chat_model?: string | null;
          chat_enabled?: boolean;
          created_at?: string;
          updated_at?: string;
        };
      };
      tasks: {
        Row: {
          id: string;
          user_id: string;
          type: 'analysis' | 'batch' | 'cleanup' | 'export';
          status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          progress: number;
          file_name: string;
          file_size: number;
          file_type: string;
          file_path: string;
          report_code: string;
          result?: {
            confidence?: number;
            is_deepfake?: boolean;
            model_name?: string;
            model_version?: string;
            summary?: Record<string, string | number | boolean | null>;
            analysis_data?: Record<string, unknown>;
            proof_json?: {
              signature?: string;
              public_key?: string;
              timestamp?: string;
              [key: string]: unknown;
            };
          };
          error?: {
            message: string;
            code?: string | number;
            details?: unknown;
            [key: string]: unknown;
          };
          started_at?: string;
          completed_at?: string;
          created_at: string;
          updated_at: string;
          deleted_at?: string;
          metadata: Record<string, unknown>;
        };
        Insert: {
          id?: string;
          user_id: string;
          type: 'analysis' | 'batch' | 'cleanup' | 'export';
          status?: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          progress?: number;
          file_name: string;
          file_size: number;
          file_type: string;
          file_path: string;
          report_code?: string;
          result?: {
            confidence?: number;
            is_deepfake?: boolean;
            model_name?: string;
            model_version?: string;
            summary?: Record<string, string | number | boolean | null>;
            analysis_data?: Record<string, unknown>;
            proof_json?: {
              signature?: string;
              public_key?: string;
              timestamp?: string;
              [key: string]: unknown;
            };
          };
          error?: {
            message: string;
            code?: string | number;
            details?: unknown;
            [key: string]: unknown;
          };
          started_at?: string;
          completed_at?: string;
          created_at?: string;
          updated_at?: string;
          deleted_at?: string;
          metadata?: Record<string, unknown>;
        };
        Update: {
          id?: string;
          user_id?: string;
          type?: 'analysis' | 'batch' | 'cleanup' | 'export';
          status?: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          progress?: number;
          file_name?: string;
          file_size?: number;
          file_type?: string;
          file_path?: string;
          report_code?: string;
          result?: {
            confidence?: number;
            is_deepfake?: boolean;
            model_name?: string;
            model_version?: string;
            summary?: Record<string, string | number | boolean | null>;
            analysis_data?: Record<string, unknown>;
            proof_json?: {
              signature?: string;
              public_key?: string;
              timestamp?: string;
              [key: string]: unknown;
            };
          } | null;
          error?: {
            message: string;
            code?: string | number;
            details?: unknown;
            [key: string]: unknown;
          } | null;
          started_at?: string | null;
          completed_at?: string | null;
          created_at?: string;
          updated_at?: string;
          deleted_at?: string | null;
          metadata?: Record<string, unknown>;
        };
      };
      notifications: {
        Row: {
          id: string;
          user_id: string;
          type: 'info' | 'success' | 'warning' | 'error' | 'scan_complete' | 'chat';
          title: string;
          message: string;
          is_read: boolean;
          action_url?: string;
          created_at: string;
          read_at?: string;
          data?: {
            task_id?: string;
            scan_id?: string;
            related_entity_type?: string;
            related_entity_id?: string;
            [key: string]: unknown;
          };
        };
        Insert: {
          id?: string;
          user_id: string;
          type?: 'info' | 'success' | 'warning' | 'error' | 'scan_complete' | 'chat';
          title: string;
          message: string;
          is_read?: boolean;
          action_url?: string;
          created_at?: string;
          read_at?: string;
          data?: {
            task_id?: string;
            scan_id?: string;
            related_entity_type?: string;
            related_entity_id?: string;
            [key: string]: unknown;
          };
        };
        Update: {
          id?: string;
          user_id?: string;
          type?: 'info' | 'success' | 'warning' | 'error' | 'scan_complete' | 'chat';
          title?: string;
          message?: string;
          is_read?: boolean;
          action_url?: string | null;
          created_at?: string;
          read_at?: string | null;
          data?: {
            task_id?: string;
            scan_id?: string;
            related_entity_type?: string;
            related_entity_id?: string;
            [key: string]: unknown;
          } | null;
        };
      };
      chat_conversations: {
        Row: {
          id: string;
          user_id: string;
          title: string;
          created_at: string;
          updated_at: string;
          is_archived: boolean;
          deleted_at?: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          title: string;
          created_at?: string;
          updated_at?: string;
          is_archived?: boolean;
          deleted_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          title?: string;
          created_at?: string;
          updated_at?: string;
          is_archived?: boolean;
          deleted_at?: string | null;
        };
      };
      chat_messages: {
        Row: {
          id: string;
          conversation_id: string;
          role: 'user' | 'assistant' | 'system';
          content: string;
          created_at: string;
          metadata: {
            tokens?: number;
            model?: string;
            temperature?: number;
            [key: string]: unknown;
          };
          deleted_at?: string;
        };
        Insert: {
          id?: string;
          conversation_id: string;
          role: 'user' | 'assistant' | 'system';
          content: string;
          created_at?: string;
          metadata?: {
            tokens?: number;
            model?: string;
            temperature?: number;
            [key: string]: unknown;
          };
          deleted_at?: string;
        };
        Update: {
          id?: string;
          conversation_id?: string;
          role?: 'user' | 'assistant' | 'system';
          content?: string;
          created_at?: string;
          metadata?: {
            tokens?: number;
            model?: string;
            temperature?: number;
            [key: string]: unknown;
          } | null;
          deleted_at?: string | null;
        };
      };
      file_uploads: {
        Row: {
          id: string;
          user_id: string;
          file_name: string;
          file_path: string;
          file_size: number;
          file_type: string;
          mime_type?: string;
          is_processed: boolean;
          created_at: string;
          expires_at?: string;
          deleted_at?: string;
          metadata: {
            width?: number;
            height?: number;
            duration?: number;
            format?: string;
            [key: string]: unknown;
          };
        };
        Insert: {
          id?: string;
          user_id: string;
          file_name: string;
          file_path: string;
          file_size: number;
          mime_type?: string;
          is_processed?: boolean;
          created_at?: string;
          expires_at?: string;
          deleted_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          file_name?: string;
          file_path?: string;
          file_size?: number;
          mime_type?: string | null;
          is_processed?: boolean;
          created_at?: string;
          expires_at?: string | null;
          deleted_at?: string | null;
        };
      };
      batch_jobs: {
        Row: {
          id: string;
          user_id: string;
          status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          total_items: number;
          processed_items: number;
          failed_items: number;
          metadata: {
            input_files?: string[];
            output_dir?: string;
            options?: {
              [key: string]: string | number | boolean | null;
            };
            [key: string]: unknown;
          };
          created_at: string;
          updated_at: string;
          completed_at?: string;
          deleted_at?: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          status?: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          total_items: number;
          processed_items?: number;
          failed_items?: number;
          metadata?: {
            input_files?: string[];
            output_dir?: string;
            options?: {
              [key: string]: string | number | boolean | null;
            };
            [key: string]: unknown;
          };
          created_at?: string;
          updated_at?: string;
          completed_at?: string;
          deleted_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          status?: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
          total_items?: number;
          processed_items?: number;
          failed_items?: number;
          metadata?: {
            input_files?: string[];
            output_dir?: string;
            options?: {
              [key: string]: string | number | boolean | null;
            };
            [key: string]: unknown;
          } | null;
          created_at?: string;
          updated_at?: string;
          completed_at?: string | null;
          deleted_at?: string | null;
        };
      };
      api_keys: {
        Row: {
          id: string;
          user_id: string;
          name: string;
          key_hash: string;
          key_algorithm: string;
          permissions: {
            endpoints?: string[];
            methods?: ('GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH')[];
            rate_limit?: number;
            expires_in_days?: number;
            [key: string]: unknown;
          };
          is_active: boolean;
          last_used_at?: string;
          expires_at?: string;
          created_at: string;
          revoked_at?: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          name: string;
          key_hash: string;
          key_algorithm?: string;
          permissions?: {
            endpoints?: string[];
            methods?: ('GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH')[];
            rate_limit?: number;
            expires_in_days?: number;
            [key: string]: unknown;
          };
          is_active?: boolean;
          last_used_at?: string;
          expires_at?: string;
          created_at?: string;
          revoked_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          name?: string;
          key_hash?: string;
          key_algorithm?: string;
          permissions?: {
            endpoints?: string[];
            methods?: ('GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH')[];
            rate_limit?: number;
            expires_in_days?: number;
            [key: string]: unknown;
          } | null;
          is_active?: boolean;
          last_used_at?: string | null;
          expires_at?: string | null;
          created_at?: string;
          revoked_at?: string | null;
        };
      };
      audit_logs: {
        Row: {
          id: string;
          user_id?: string;
          action: string;
          table_name: string;
          row_id?: string;
          changes?: Record<string, unknown>;
          ip_address?: string;
          user_agent?: string;
          created_at: string;
        };
        Insert: {
          id?: string;
          user_id?: string;
          action?: string;
          table_name?: string;
          row_id?: string;
          changes?: Record<string, unknown>;
          ip_address?: string;
          user_agent?: string;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string | null;
          action?: string;
          table_name?: string;
          row_id?: string | null;
          changes?: Record<string, unknown> | null;
          ip_address?: string | null;
          user_agent?: string | null;
          created_at?: string;
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
      user_role: 'user' | 'admin' | 'moderator';
      analysis_status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
      media_type: 'image' | 'video' | 'audio' | 'multimodal' | 'batch';
      notification_type: 'info' | 'success' | 'warning' | 'error' | 'scan_complete' | 'chat';
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
};
