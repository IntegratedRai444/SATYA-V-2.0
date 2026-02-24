const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '.env' });

// Use admin client for creation (like in the code)
const { createClient: createAdminClient } = require('@supabase/supabase-js');
const supabaseAdmin = createAdminClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function testJobCreation() {
  console.log('Testing job creation process...');
  
  const userId = 'test-user-id';
  const jobId = '412aae41-e6f4-4b19-8851-84044574ae8f'; // The failing jobId
  const reportCode = `RPT-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
  
  console.log('Creating job with:', { userId, jobId, reportCode });
  
  // Exact same logic as createAnalysisJob
  const { data, error } = await supabaseAdmin
    .from('tasks')
    .insert({
      id: jobId, // Use the failing jobId
      user_id: userId,
      type: 'analysis',
      status: 'processing',
      file_name: 'test.jpg',
      file_type: 'image/jpeg',
      file_size: 12345,
      report_code: reportCode,
      metadata: {
        media_type: 'image'
      },
      progress: 0,
      started_at: new Date().toISOString(),
      result: {
        confidence: 0,
        is_deepfake: false,
        model_name: 'SatyaAI',
        model_version: '1.0.0',
        summary: {}
      }
    })
    .select('id, report_code')
    .single();

  console.log('Insert result:', { data, error });
  
  if (data) {
    console.log('✅ Job created successfully:', data.id);
    
    // Now test if we can find it
    const { data: foundJob, error: findError } = await supabaseAdmin
      .from('tasks')
      .select('*')
      .eq('id', jobId)
      .single();
      
    console.log('Find result:', { found: !!foundJob, error: findError });
    if (foundJob) {
      console.log('Found job details:', foundJob.id, foundJob.status, foundJob.user_id);
    }
  } else {
    console.log('❌ Job creation failed:', error);
  }
}

testJobCreation().catch(console.error);
