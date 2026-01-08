import { User } from '@supabase/supabase-js';

export type SupabaseUser = User & {
  role?: string;
  email_verified?: boolean;
  phone_verified?: boolean;
  user_metadata?: Record<string, any>;
  app_metadata?: {
    provider?: string;
    [key: string]: any;
  };
};

export interface AuthResponse {
  valid: boolean;
  user?: SupabaseUser;
  error?: string;
}
