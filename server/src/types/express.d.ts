import { User } from '@supabase/supabase-js';

declare global {
  namespace Express {
    interface Request {
      id?: string;
      user?: User & {
        email_verified: boolean;
        role?: string;
        user_metadata?: Record<string, unknown>;
      };
    }
  }
}

export {};
