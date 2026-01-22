import { createClient } from '@supabase/supabase-js';

// Test Supabase anon key connection
const supabaseUrl = 'https://ftbpbghcebwgzqfsgmxk.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ0YnBiZ2hjZWJ3Z3pxZnNnbXhrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgwNTc2MTcsImV4cCI6MjA4MzYzMzYxN30.JFbrt84nne3v2gPfhWiixsd5fWbkVkEWFGC2P2KoqUc';

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testAnonKey() {
  console.log('Testing Supabase anon key connection...\n');
  
  // Test 1: Basic connection test
  console.log('1. Testing basic connection...');
  try {
    const { data, error } = await supabase
      .from('users')
      .select('count')
      .limit(1);
    
    if (error) {
      console.error('❌ Connection failed:', error.message);
    } else {
      console.log('✅ Connection successful');
      console.log('   Data:', data);
    }
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
  
  // Test 2: Test RLS policies (try to read all users)
  console.log('\n2. Testing Row Level Security (RLS)...');
  try {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .limit(5);
    
    if (error) {
      console.error('❌ RLS test failed:', error.message);
      if (error.code === 'PGRST116') {
        console.log('   This might mean the table doesn\'t exist or RLS is blocking access');
      }
    } else {
      console.log('✅ RLS test passed');
      console.log(`   Found ${data.length} users (this should be 0 or limited by RLS)`);
    }
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
  
  // Test 3: Test public tables (if any)
  console.log('\n3. Testing access to public information...');
  try {
    const { data, error } = await supabase
      .rpc('get_public_info') // Try to call a public function if it exists
      .limit(1);
    
    if (error) {
      console.log('ℹ️  No public RPC function found (this is normal)');
    } else {
      console.log('✅ Public RPC function accessible');
      console.log('   Data:', data);
    }
  } catch (err) {
    console.log('ℹ️  No public RPC function found (this is normal)');
  }
  
  // Test 4: Test authentication endpoint
  console.log('\n4. Testing authentication endpoints...');
  try {
    const { data, error } = await supabase.auth.getSession();
    
    if (error) {
      console.error('❌ Auth test failed:', error.message);
    } else {
      console.log('✅ Auth endpoint accessible');
      console.log('   Current session:', data.session ? 'Active' : 'None (expected for anon)');
    }
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
  
  console.log('\n✅ Anon key test completed!');
}

// Run the test
testAnonKey().catch(console.error);
