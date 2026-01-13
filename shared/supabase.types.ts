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
      // Add your table types here based on your database schema
      // Example:
      users: {
        Row: {
          id: string
          created_at: string
          email: string
          // Add other user fields
        }
        Insert: {
          id?: string
          created_at?: string
          email: string
          // Add other user fields
        }
        Update: {
          id?: string
          created_at?: string
          email?: string
          // Add other user fields
        }
      }
      // Add other tables as needed
    }
    Views: {
      // Add your view types here if any
      [_ in never]: never
    }
    Functions: {
      // Add your function types here if any
      execute_sql: {
        Args: {
          query: string
          params: Json[]
        }
        Returns: any
      }
    }
    Enums: {
      // Add your enum types here if any
      [_ in never]: never
    }
  }
}

// Helper types
export type Tables<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Row']
export type Enums<T extends keyof Database['public']['Enums']> = Database['public']['Enums'][T]
