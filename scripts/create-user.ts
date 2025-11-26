import 'dotenv/config';
import bcrypt from 'bcrypt';
import { db } from '../server/db';
import { users } from '../shared/schema';
import { eq } from 'drizzle-orm';

async function createUser() {
  try {
    const username = 'rishabhkapoor';
    const email = 'rishabhkapoor@atomicmail.io';
    const password = 'Rishabhkapoor@0444';
    const fullName = 'Rishabh Kapoor';

    console.log('🔐 Creating user...');
    console.log(`Username: ${username}`);
    console.log(`Email: ${email}`);

    // Check if user already exists
    const existingUser = await db.select()
      .from(users)
      .where(eq(users.username, username))
      .limit(1);

    if (existingUser.length > 0) {
      console.log('⚠️  User already exists. Updating password...');
      
      // Hash new password
      const saltRounds = 12;
      const hashedPassword = await bcrypt.hash(password, saltRounds);
      
      // Update user
      await db.update(users)
        .set({ 
          password: hashedPassword,
          email: email,
          fullName: fullName,
          updatedAt: new Date()
        })
        .where(eq(users.username, username));
      
      console.log('✅ User password updated successfully!');
    } else {
      // Hash password
      const saltRounds = 12;
      const hashedPassword = await bcrypt.hash(password, saltRounds);

      // Create user
      const [newUser] = await db.insert(users).values({
        username: username,
        password: hashedPassword,
        email: email,
        fullName: fullName,
        role: 'user'
      }).returning();

      console.log('✅ User created successfully!');
      console.log(`User ID: ${newUser.id}`);
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
