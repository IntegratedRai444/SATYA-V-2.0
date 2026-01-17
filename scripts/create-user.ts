import 'dotenv/config';
import bcrypt from 'bcrypt';
import { db } from '../server/db';
import { supabase } from '../server/config/supabase';

async function createUser() {
  try {
    const username = 'rishabhkapoor';
    const email = 'rishabhkapoor@atomicmail.io';
    const password = 'Rishabhkapoor@0444';
    const fullName = 'Rishabh Kapoor';

    console.log('🔐 Creating user...');
    console.log(`Username: ${username}`);
    console.log(`Email: ${email}`);

    // Check if user already exists in Supabase auth
    const { data: existingUser, error: checkError } = await supabase
      .from('users')
      .select('*')
      .eq('username', username)
      .limit(1);

    if (checkError) {
      console.error('Error checking existing user:', checkError);
      return;
    }

    if (existingUser && existingUser.length > 0) {
      console.log('⚠️  User already exists. Updating password...');
      
      // Hash new password
      const saltRounds = 12;
      const hashedPassword = await bcrypt.hash(password, saltRounds);
      
      // Update user in Supabase
      const { error: updateError } = await supabase
        .from('users')
        .update({ 
          email: email,
          full_name: fullName,
          updated_at: new Date().toISOString()
        })
        .eq('username', username);
      
      if (updateError) {
        console.error('Error updating user:', updateError);
        return;
      }
      
      console.log('✅ User updated successfully!');
    } else {
      // Create user in Supabase auth first
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email: email,
        password: password,
        options: {
          data: {
            username: username,
            full_name: fullName,
            role: 'user'
          }
        }
      });

      if (authError) {
        console.error('Error creating auth user:', authError);
        return;
      }

      console.log('✅ User created successfully!');
      console.log(`User ID: ${authData.user?.id}`);
    }

    console.log('\n📋 Login credentials:');
    console.log(`Username: ${username}`);
    console.log(`Email: ${email}`);
    console.log(`Password: ${password}`);
    console.log('\n✨ You can now login with these credentials!');

    process.exit(0);
  } catch (error) {
    console.error('❌ Error creating user:', error);
    process.exit(1);
  }
}

createUser();
