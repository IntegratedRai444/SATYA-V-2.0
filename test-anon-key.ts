import { createClient } from '@supabase/supabase-js';

// Test Supabase anon key connection
const supabaseUrl = 'https://ftbpbghcebwgzqfsgmxk.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ0YnBiZ2hjZWJ3Z3pxZnNnbXhrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgwNTc2MTcsImV4cCI6MjA4MzYzMzYxN30.JFbrt84nne3v2gPfhWiixsd5fWbkVkEWFGC2P2KoqUc';

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testAnonKey() {
  // Test 1: Basic connection test
  try {
    const { data, error } = await supabase
      .from('users')
      .select('count')
      .limit(1);
    
    if (error) {
          } else {
          }
  } catch (err) {
      }
  
  // Test 2: Test RLS policies (try to read all users)
    try {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .limit(5);
    
    if (error) {
          }
  } catch (err) {
      }
  
  // Test 3: Test public tables (if any)
    try {
    const { data, error } = await supabase
      .rpc('get_public_info') // Try to call a public function if it exists
      .limit(1);
    
    if (error) {
          }
  } catch (err) {
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
      }
  
  console.log('\n✅ Anon key test completed!');
}

// Run the test
testAnonKey().catch(console.error);
