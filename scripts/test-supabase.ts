import 'dotenv/config';
import { supabase, testSupabaseConnection } from '../server/services/supabase-client';

async function testSupabase() {
  console.log('🔍 Testing Supabase connection...\n');
  
  // Test connection
  const connected = await testSupabaseConnection();
  
  if (!connected) {
    console.log('\n❌ Supabase connection failed');
    console.log('Please check:');
    console.log('1. Your Supabase project is active (not paused)');
    console.log('2. SUPABASE_URL and SUPABASE_ANON_KEY are correct in .env');
    console.log('3. The users table exists in your database');
    process.exit(1);
  }
  
  console.log('\n✅ Supabase connection successful!');
  
  // Try to create a test user
  console.log('\n📝 Creating test user...');
  
  const { data, error } = await supabase
    .from('users')
    .insert({
      username: 'rishabhkapoor',
      password: '$2b$12$test', // This will be replaced with proper hash
      email: 'rishabhkapoor@atomicmail.io',
      full_name: 'Rishabh Kapoor',
      role: 'user',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    })
    .select()
    .single();
  
  if (error) {
    if (error.code === '23505') {
      console.log('⚠️  User already exists');
    } else {
      console.log('❌ Error creating user:', error.message);
      console.log('Error code:', error.code);
    }
  } else {
    console.log('✅ Test user created:', data);
  }
  
  process.exit(0);
}

testSupabase();
