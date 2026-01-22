import { SupabaseUser } from './supabase';

declare global {
  namespace Express {
    interface Request {
      id?: string;
      user?: SupabaseUser & {
        id: string;
        role: string;
        email: string;
        user_metadata?: Record<string, unknown>;
      };
      file?: Express.Multer.File;
    }
  }
}

// This empty export makes this file a module
export {};
