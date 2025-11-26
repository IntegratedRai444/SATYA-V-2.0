import { createClient } from '@supabase/supabase-js';
import { logger } from '../config';

// Validate required environment variables
if (!process.env.SUPABASE_URL) {
  throw new Error('SUPABASE_URL environment variable is required');
}
if (!process.env.SUPABASE_ANON_KEY) {
  throw new Error('SUPABASE_ANON_KEY environment variable is required');
}

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseKey);

// Test connection
export async function testSupabaseConnection(): Promise<boolean> {
  try {
    const { error } = await supabase.from('users').select('count').limit(1);
    if (error) {
      logger.warn('⚠️  Supabase connection test failed', { error: error.message });
      return false;
    }
    logger.info('✅ Supabase client connected successfully');
    return true;
  } catch (error) {
    logger.warn('⚠️  Supabase connection test error', { 
      error: (error as Error).message 
    });
    return false;
  }
}
