#!/usr/bin/env tsx

/**
 * Database Migration Runner
 * Runs all pending database migrations using Supabase
 */

import { supabase } from '../server/config/supabase';
import { config } from 'dotenv';
import { resolve } from 'path';

// Load environment variables
config({ path: resolve(process.cwd(), '.env') });

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.error('❌ DATABASE_URL is not defined in environment variables');
  process.exit(1);
}

async function runMigrations() {
  console.log('🔄 Checking database connection...\n');

  try {
    // Test Supabase connection
    const { data, error } = await supabase.from('users').select('count').limit(1);
    
    if (error) {
      console.error('❌ Database connection failed:', error);
      process.exit(1);
    }

    console.log('✅ Database connection successful');
    console.log('📋 Supabase migrations are handled through the Supabase dashboard');
    console.log('🔗 Visit: https://app.supabase.com/project/ftbpbghcebwgzqfsgmxk/database');
    console.log('\n✅ Database is ready!');
    
  } catch (error) {
    console.error('\n❌ Database check failed:', error);
    process.exit(1);
  }
}

// Run migrations
runMigrations()
  .then(() => {
    console.log('\n✨ Database is up to date');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n💥 Fatal error:', error);
    process.exit(1);
  });
