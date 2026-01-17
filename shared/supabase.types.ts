export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      users: {
        Row: {
          id: number
          username: string
          password: string
          email: string | null
          full_name: string | null
          api_key: string | null
          role: string
          failed_login_attempts: number
          last_failed_login: string | null
          is_locked: boolean
          lockout_until: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: number
          username: string
          password: string
          email?: string | null
          full_name?: string | null
          api_key?: string | null
          role?: string
          failed_login_attempts?: number
          last_failed_login?: string | null
          is_locked?: boolean
          lockout_until?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: number
          username?: string
          password?: string
          email?: string | null
          full_name?: string | null
          api_key?: string | null
          role?: string
          failed_login_attempts?: number
          last_failed_login?: string | null
          is_locked?: boolean
          lockout_until?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      scans: {
        Row: {
          id: number
          user_id: number
          filename: string
          type: string
          result: string
          confidence_score: number
          detection_details: Json | null
          metadata: Json | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: number
          user_id: number
          filename: string
          type: string
          result: string
          confidence_score: number
          detection_details?: Json | null
          metadata?: Json | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: number
          user_id?: number
          filename?: string
          type?: string
          result?: string
          confidence_score?: number
          detection_details?: Json | null
          metadata?: Json | null
          created_at?: string
          updated_at?: string
        }
      }
      user_preferences: {
        Row: {
          id: number
          user_id: number
          theme: string
          language: string
          confidence_threshold: number
          enable_notifications: boolean
          auto_analyze: boolean
          sensitivity_level: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: number
          user_id: number
          theme?: string
          language?: string
          confidence_threshold?: number
          enable_notifications?: boolean
          auto_analyze?: boolean
          sensitivity_level?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: number
          user_id?: number
          theme?: string
          language?: string
          confidence_threshold?: number
          enable_notifications?: boolean
          auto_analyze?: boolean
          sensitivity_level?: string
          created_at?: string
          updated_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
  }
}
