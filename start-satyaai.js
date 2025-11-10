#!/usr/bin/env node

/**
 * SatyaAI Startup Script
 * Starts both Node.js and Python servers with proper coordination
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const config = {
  nodePort: process.env.PORT || 3000,
  pythonPort: process.env.PYTHON_SERVER_PORT || 5001,
  nodeEnv: process.env.NODE_ENV || 'development'
};

console.log('🚀 Starting SatyaAI Detection System...');
console.log('='.repeat(60));
console.log(`Environment: ${config.nodeEnv}`);
console.log(`Node.js Server Port: ${config.nodePort}`);
console.log(`Python Server Port: ${config.pythonPort}`);
console.log('='.repeat(60));

// Track running processes
let nodeProcess = null;
let pythonProcess = null;
let clientProcess = null;

// Cleanup function
function cleanup() {
  console.log('\n🛑 Shutting down SatyaAI...');

  if (nodeProcess) {
    console.log('Stopping Node.js server...');
    nodeProcess.kill('SIGTERM');
  }

  if (pythonProcess) {
    console.log('Stopping Python server...');
    pythonProcess.kill('SIGTERM');
  }

  if (clientProcess) {
    console.log('Stopping client development server...');
    clientProcess.kill('SIGTERM');
  }

  setTimeout(() => {
    console.log('✅ SatyaAI shutdown complete');
    process.exit(0);
  }, 2000);
}

// Handle shutdown signals
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
process.on('exit', cleanup);

// Start Python server
function startPythonServer() {
  return new Promise((resolve, reject) => {
    console.log('🐍 Starting Python AI server...');

    const pythonScript = path.join(__dirname, 'run_satyaai.py');

    if (!fs.existsSync(pythonScript)) {
      console.error('❌ Python startup script not found:', pythonScript);
      reject(new Error('Python script not found'));
      return;
    }

    pythonProcess = spawn('python', [pythonScript], {
      stdio: 'pipe',
      env: {
        ...process.env,
        PORT: config.pythonPort,
        NODE_ENV: config.nodeEnv,
        PYTHON_SERVER_PORT: config.pythonPort
      }
    });

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log(`[Python] ${output}`);

        // Check if server is ready
        if (output.includes('Running on') || output.includes('Server running')) {
          resolve();
        }
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error && !error.includes('WARNING')) {
        console.error(`[Python Error] ${error}`);
      }
    });

    pythonProcess.on('exit', (code) => {
      console.log(`Python server exited with code: ${code}`);
      pythonProcess = null;
    });

    pythonProcess.on('error', (err) => {
      console.error('Failed to start Python server:', err.message);
      reject(err);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (pythonProcess && !pythonProcess.killed) {
        console.log('✅ Python server started (timeout reached, assuming success)');
        resolve();
      }
    }, 30000);
  });
}

// Start Node.js server
function startNodeServer() {
  return new Promise((resolve, reject) => {
    console.log('🟢 Starting Node.js server...');

    const nodeScript = config.nodeEnv === 'production' ? 'start' : 'dev';

    nodeProcess = spawn('npm', ['run', nodeScript], {
      stdio: 'pipe',
      shell: true,
      env: { ...process.env, PORT: config.nodePort }
    });

    nodeProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log(`[Node.js] ${output}`);

        // Check if server is ready
        if (output.includes('Server running') || output.includes('ready')) {
          resolve();
        }
      }
    });

    nodeProcess.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error && !error.includes('WARNING')) {
        console.error(`[Node.js Error] ${error}`);
      }
    });

    nodeProcess.on('exit', (code) => {
      console.log(`Node.js server exited with code: ${code}`);
      nodeProcess = null;
    });

    nodeProcess.on('error', (err) => {
      console.error('Failed to start Node.js server:', err.message);
      reject(err);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (nodeProcess && !nodeProcess.killed) {
        console.log('✅ Node.js server started');
        resolve();
      }
    }, 30000);
  });
}

// Start client development server (only in development)
function startClientServer() {
  if (config.nodeEnv === 'production') {
    console.log('📦 Production mode: Client will be served by Node.js server');
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    console.log('⚛️  Starting React development server...');

    clientProcess = spawn('npm', ['run', 'dev:client'], {
      stdio: 'pipe',
      shell: true
    });

    clientProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log(`[React] ${output}`);

        // Check if server is ready
        if (output.includes('Local:') || output.includes('ready')) {
          resolve();
        }
      }
    });

    clientProcess.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error && !error.includes('WARNING')) {
        console.error(`[React Error] ${error}`);
      }
    });

    clientProcess.on('exit', (code) => {
      console.log(`React server exited with code: ${code}`);
      clientProcess = null;
    });

    clientProcess.on('error', (err) => {
      console.error('Failed to start React server:', err.message);
      reject(err);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (clientProcess && !clientProcess.killed) {
        console.log('✅ React development server started');
        resolve();
      }
    }, 30000);
  });
}

// Main startup sequence
async function startSatyaAI() {
  try {
    // Start Python server first
    await startPythonServer();

    // Start Node.js server
    await startNodeServer();

    // Start client server (development only)
    await startClientServer();

    console.log('\n🎉 SatyaAI is now running!');
    console.log('='.repeat(60));

    if (config.nodeEnv === 'development') {
      console.log(`🌐 Frontend: http://localhost:5173`);
      console.log(`🔧 Backend API: http://localhost:${config.nodePort}`);
      console.log(`🐍 Python AI: http://localhost:${config.pythonPort}`);
    } else {
      console.log(`🌐 Application: http://localhost:${config.nodePort}`);
    }

    console.log('='.repeat(60));
    console.log('Press Ctrl+C to stop all servers');

    // Keep the process alive
    process.stdin.resume();

  } catch (error) {
    console.error('❌ Failed to start SatyaAI:', error.message);
    cleanup();
    process.exit(1);
  }
}

// Check if required files exist
function checkRequirements() {
  const requiredFiles = [
    'package.json',
    'server/index.ts',
    'run_satyaai.py'
  ];

  for (const file of requiredFiles) {
    if (!fs.existsSync(file)) {
      console.error(`❌ Required file not found: ${file}`);
      process.exit(1);
    }
  }

  console.log('✅ All required files found');
}

// Start the application
checkRequirements();
startSatyaAI();