require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase configuration');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function testConnection() {
  try {
    console.log('🔌 Testing Supabase connection...');
    
    // Simple query to test connection
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .limit(1);
      
    if (error) throw error;
    
    console.log('✅ Successfully connected to Supabase!');
    console.log('📊 First profile:', data[0] || 'No profiles found');
    
  } catch (error) {
    console.error('❌ Connection failed:', error.message);
  }
}

testConnection();
