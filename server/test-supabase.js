require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

(async () => {
  try {
    console.log(' Testing Supabase connection...');
    
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY
    );

    // Test auth
    const { data: { session }, error: authError } = await supabase.auth.getSession();
    if (authError) throw authError;
    
    console.log(' Successfully connected to Supabase Auth');
    console.log( Session active: );
    
    // Test database
    const { data, error: dbError } = await supabase
      .from('profiles')
      .select('*')
      .limit(1);
      
    if (dbError) throw dbError;
    
    console.log( Found  profiles);
    
  } catch (error) {
    console.error(' Error:', error.message);
    console.log('\nTroubleshooting:');
    console.log('1. Check if Supabase URL and keys are correct');
    console.log('2. Verify the table names in your database');
    console.log('3. Check your internet connection');
  }
})();
