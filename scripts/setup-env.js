#!/usr/bin/env node

/**
 * Interactive script to help set up .env file
 * Run: node scripts/setup-env.js
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(query) {
  return new Promise(resolve => rl.question(query, resolve));
}

function generateSecret(length = 32) {
  return crypto.randomBytes(length).toString('hex');
}

async function main() {
  console.log('\n🔐 SatyaAI Environment Setup\n');
  console.log('This script will help you create a secure .env file.\n');

  const envPath = path.join(__dirname, '..', '.env');
  
  // Check if .env already exists
  if (fs.existsSync(envPath)) {
    const overwrite = await question('.env file already exists. Overwrite? (y/N): ');
    if (overwrite.toLowerCase() !== 'y') {
      console.log('\n✅ Setup cancelled. Existing .env file preserved.\n');
      rl.close();
      return;
    }
  }

  console.log('\n📝 Please provide the following information:\n');

  // Supabase Configuration
  console.log('--- Supabase Configuration ---');
  const supabaseUrl = await question('Supabase URL: ');
  const supabaseAnonKey = await question('Supabase Anon Key: ');
  const databaseUrl = await question('Database URL (PostgreSQL connection string): ');

  // Generate secure secrets
  console.log('\n--- Generating Secure Secrets ---');
  const jwtSecret = generateSecret(32);
  const jwtSecretKey = generateSecret(32);
  const sessionSecret = generateSecret(32);
  const flaskSecretKey = generateSecret(32);
  
  console.log('✅ Generated secure random secrets');

  // Optional configurations
  console.log('\n--- Optional Configuration ---');
  const nodeEnv = await question('Environment (development/production) [development]: ') || 'development';
  const port = await question('Server Port [3000]: ') || '3000';
  const pythonApiUrl = await question('Python API URL [http://localhost:5001]: ') || 'http://localhost:5001';
  
  // OpenAI (optional)
  console.log('\n--- OpenAI Configuration (Optional - for AI Assistant) ---');
  const openaiApiKey = await question('OpenAI API Key (press Enter to skip): ');

  // Build .env content
  const envContent = `# SatyaAI Environment Configuration
# Generated on ${new Date().toISOString()}
# NEVER commit this file to version control!

# ============================================================================
# REQUIRED - Application Configuration
# ============================================================================

# Supabase Configuration
SUPABASE_URL=${supabaseUrl}
SUPABASE_ANON_KEY=${supabaseAnonKey}

# Database Connection
DATABASE_URL=${databaseUrl}

# JWT Authentication (Auto-generated secure secrets)
JWT_SECRET=${jwtSecret}
JWT_SECRET_KEY=${jwtSecretKey}
JWT_EXPIRES_IN=24h

# Session Management (Auto-generated secure secret)
SESSION_SECRET=${sessionSecret}

# Flask/Python Server (Auto-generated secure secret)
FLASK_SECRET_KEY=${flaskSecretKey}

# ============================================================================
# OPTIONAL - Has defaults
# ============================================================================

# Server Configuration
NODE_ENV=${nodeEnv}
PORT=${port}
FLASK_HOST=0.0.0.0
FLASK_ENV=${nodeEnv}

# Python AI Server
PYTHON_API_URL=${pythonApiUrl}

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173

# Redis (Optional - for production caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

${openaiApiKey ? `# OpenAI Configuration
OPENAI_API_KEY=${openaiApiKey}
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=2000
` : '# OpenAI Configuration (Not configured)\n# OPENAI_API_KEY=\n# OPENAI_MODEL=gpt-4o-mini\n# OPENAI_MAX_TOKENS=2000\n'}
# File Upload Configuration
MAX_FILE_SIZE=104857600

# Detection Configuration
CONFIDENCE_THRESHOLD=0.7
FACE_DETECTION_THRESHOLD=0.8

# Performance Configuration
BATCH_SIZE=8
NUM_WORKERS=4
CACHE_SIZE=1000

# Logging
LOG_LEVEL=INFO
`;

  // Write .env file
  fs.writeFileSync(envPath, envContent, 'utf8');

  console.log('\n✅ .env file created successfully!\n');
  console.log('📋 Summary:');
  console.log(`   - Supabase URL: ${supabaseUrl}`);
  console.log(`   - Environment: ${nodeEnv}`);
  console.log(`   - Port: ${port}`);
  console.log(`   - Secure secrets: Generated automatically`);
  console.log(`   - OpenAI: ${openaiApiKey ? 'Configured' : 'Not configured'}`);
  console.log('\n⚠️  IMPORTANT: Never commit .env to version control!');
  console.log('   The .env file contains sensitive credentials.\n');

  rl.close();
}

main().catch(error => {
  console.error('\n❌ Error:', error.message);
  rl.close();
  process.exit(1);
});
