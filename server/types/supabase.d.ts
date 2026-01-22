import { User } from '@supabase/supabase-js';

export type SupabaseUser = User & {
  role?: string;
  email_verified?: boolean;
  phone_verified?: boolean;
  user_metadata?: Record<string, unknown>;
  app_metadata?: {
    provider?: string;
    [key: string]: unknown;
  };
};

export interface AuthResponse {
  valid: boolean;
  user?: SupabaseUser;
  error?: string;
}
