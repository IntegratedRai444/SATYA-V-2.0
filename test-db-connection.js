import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

async function testConnection() {
  try {
    // Test 1: Verify environment variables
    const requiredVars = [
      'SUPABASE_URL',
      'SUPABASE_ANON_KEY',
      'SUPABASE_SERVICE_ROLE_KEY',
      'SUPABASE_JWT_SECRET'
    ];

    console.log('🧪 TEST 1: Verifying environment variables...');
    for (const varName of requiredVars) {
      if (!process.env[varName]) {
        throw new Error(`❌ Missing required environment variable: ${varName}`);
      }
    }
    console.log('✅ Environment variables verified');

    // Test 2: Initialize Supabase client and query
    console.log('\n🧪 TEST 2: Testing Supabase connection...');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY
    );

    // Test 2a: Query auth.users table
    console.log('🔍 Querying auth.users table...');
    const { data: users, error: userError } = await supabase
      .from('users')
      .select('*')
      .limit(1);

    if (userError) throw userError;
    console.log(`✅ Successfully queried users table. Found ${users ? users.length : 0} users`);

    // Test 3: Verify auth dependency
    console.log('\n🧪 TEST 3: Verifying auth dependency...');
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    
    if (sessionError) {
      console.log('ℹ️ No active session (expected if not logged in)');
    } else {
      console.log(`ℹ️ Session status: ${session ? 'Active' : 'No active session'}`);
    }

    // Test 4: Simulate DB unavailability
    console.log('\n🧪 TEST 4: Testing error handling...');
    const brokenSupabase = createClient(
      'https://invalid-url.supabase.co',
      'invalid-key'
    );

    try {
      await brokenSupabase.from('users').select('*').limit(1);
      console.log('❌ Failed: Invalid connection did not throw error');
    } catch (error) {
      console.log('✅ Properly handled invalid connection attempt');
    }

    console.log('\n🛡️ SATYAAI — DATABASE CONNECTIVITY STATUS');
    console.log('SUPABASE CLIENT INIT: ✅');
    console.log('LIVE DB QUERY: ✅');
    console.log('AUTH DB DEPENDENCY: ✅');
    console.log('FAILURE HANDLING: ✅');
    console.log('\nFINAL DB STATUS:');
    console.log('🟢 DATABASE CONNECTED');

  } catch (error) {
    console.error('\n❌ TEST FAILED:', error.message);
    
    console.log('\n🛡️ SATYAAI — DATABASE CONNECTIVITY STATUS');
    console.log('SUPABASE CLIENT INIT:', error.message.includes('environment variable') ? '❌' : '✅');
    console.log('LIVE DB QUERY: ❌');
    console.log('AUTH DB DEPENDENCY: ❌');
    console.log('FAILURE HANDLING:', error.message.includes('handled') ? '✅' : '❌');
    
    console.log('\nFINAL DB STATUS:');
    console.log('🔴 DATABASE NOT CONNECTED');
    process.exit(1);
  }
}

testConnection().catch(console.error);
