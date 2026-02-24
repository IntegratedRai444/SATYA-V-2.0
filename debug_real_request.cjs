const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '.env' });

// Simulate the exact same process as analysis route
async function debugRequest() {
  console.log('=== DEBUGGING REAL REQUEST ===');
  
  // Get a real user ID from auth (simulate)
  const { data: { users } } = await createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  )
    .from('auth.users')
    .select('id')
    .limit(1);
    
  if (!users || users.length === 0) {
    console.log('❌ No users found in auth system');
    return;
  }
  
  const realUserId = users[0].id;
  console.log('✅ Found real user ID:', realUserId);
  
  // Test createAnalysisJob with real user ID
  const { createAnalysisJob } = require('./server/routes/history.ts');
  const jobId = '412aae41-e6f4-4b19-8851-84044574ae8f';
  
  try {
    const job = await createAnalysisJob(realUserId, {
      modality: 'image',
      filename: 'test.jpg',
      mime_type: 'image/jpeg',
      size_bytes: 12345,
      metadata: {
        originalName: 'test.jpg',
        mimeType: 'image/jpeg',
        size: 12345,
      },
    }, jobId);
    
    console.log('✅ createAnalysisJob succeeded:', job);
  } catch (error) {
    console.log('❌ createAnalysisJob failed:', error);
  }
}

debugRequest().catch(console.error);
