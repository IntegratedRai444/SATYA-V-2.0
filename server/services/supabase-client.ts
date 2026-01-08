import { createClient } from '@supabase/supabase-js';
import { config } from '../config/environment';
import { logger } from '../config';

// Use the centralized configuration
const supabaseUrl = config.SUPABASE_URL;
const supabaseKey = config.SUPABASE_ANON_KEY;

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
