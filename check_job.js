const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '.env' });

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

async function checkJob() {
  const jobId = '412aae41-e6f4-4b19-8851-84044574ae8f';
  console.log('Checking for job:', jobId);
  
  // Check if job exists without user filter
  const { data: job, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('id', jobId)
    .single();
    
  console.log('Job exists:', !!job);
  console.log('Error:', error);
  if (job) {
    console.log('Job status:', job.status);
    console.log('Job user_id:', job.user_id);
    console.log('Job type:', job.type);
    console.log('Job created_at:', job.created_at);
  }
  
  // Now test the exact query from results route
  console.log('\n--- Testing Results Route Query ---');
  const userId = job?.user_id;
  if (userId) {
    const { data: task, error: taskError } = await supabase
      .from('tasks')
      .select(`id, type, status, file_name, file_type, file_size, result, report_code, created_at, completed_at, error`)
      .eq('id', jobId)
      .eq('user_id', userId)
      .eq('type', 'analysis')
      .single();
      
    console.log('Results query success:', !!task);
    console.log('Results query error:', taskError);
    if (task) {
      console.log('Task found via results query:', task.id, task.status);
    }
  }
}

checkJob().catch(console.error);
