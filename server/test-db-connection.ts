import { createClient } from '@supabase/supabase-js';
import { config } from 'dotenv';
import { logger } from './config/logger';

// Load environment variables
config();

async function testConnection() {
  try {
    // Test direct PostgreSQL connection
    logger.info('🔍 Testing database connection...');
    
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Missing Supabase URL or Service Role Key in environment variables');
    }

    // Create a Supabase client
    const supabase = createClient(supabaseUrl, supabaseKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    });

    // First, try to list tables using information_schema (standard SQL)
    logger.info('🔍 Listing available tables...');
    
    // Try to connect and list tables using a direct query
    const { data: tables, error } = await supabase
      .rpc('get_tables')
      .select('*')
      .limit(5);
    
    if (error) {
      logger.warn('⚠️ Could not list tables directly, trying alternative approach...');
      
      // If the first approach fails, try to query a specific table
      const { data: testData, error: testError } = await supabase
        .from('user_statistics')
        .select('*')
        .limit(1);
        
      if (testError) {
        throw new Error(`Could not query tables. This might be due to Row Level Security (RLS) or missing permissions. Error: ${testError.message}`);
      }
      
      logger.info('✅ Successfully connected to Supabase database');
      logger.info('ℹ️  Successfully queried user_statistics table');
    } else {
      logger.info('✅ Successfully connected to Supabase database');
      if (tables && tables.length > 0) {
        logger.info('📋 Available tables:', tables.map((t: any) => t.tablename).join(', '));
      } else {
        logger.info('ℹ️  No tables found or insufficient permissions to list tables');
      }
    }
    
    process.exit(0);
  } catch (error) {
    logger.error('❌ Database connection failed:', error);
    process.exit(1);
  }
}

testConnection();
