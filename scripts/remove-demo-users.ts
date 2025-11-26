import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://cvuweocfdirkbelojzwr.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2dXdlb2NmZGlya2JlbG9qendyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3MDg3OTIsImV4cCI6MjA3ODI4NDc5Mn0.lj_RfH6jUf6Mzw1xXpyPN32jkX7meLam-iGvY5d6sNo';

async function removeDemoUsers() {
  console.log('🗑️  Removing demo and admin users...\n');
  
  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  
  const usersToRemove = ['demo', 'admin'];
  
  for (const username of usersToRemove) {
    console.log(`Removing ${username}...`);
    
    const { error } = await supabase
      .from('users')
      .delete()
      .eq('username', username);
    
    if (error) {
      console.log(`  ❌ Failed: ${error.message}`);
    } else {
      console.log(`  ✅ Removed`);
    }
  }
  
  console.log('\n✅ Done! Only rishabhkapoor account remains.');
  console.log('\nYou can now login with:');
  console.log('  Username: rishabhkapoor');
  console.log('  Password: Rishabhkapoor@0444');
}

removeDemoUsers();
