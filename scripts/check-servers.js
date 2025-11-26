#!/usr/bin/env node

/**
 * Check if all servers are running
 * Run: node scripts/check-servers.js
 */

const http = require('http');

const servers = [
  { name: 'Node.js Backend', url: 'http://localhost:3000/health', port: 3000 },
  { name: 'Python AI Server', url: 'http://localhost:5001/health', port: 5001 },
  { name: 'React Frontend', url: 'http://localhost:5173', port: 5173 }
];

function checkServer(server) {
  return new Promise((resolve) => {
    const url = new URL(server.url);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'GET',
      timeout: 3000
    };

    const req = http.request(options, (res) => {
      resolve({
        ...server,
        status: 'running',
        statusCode: res.statusCode
      });
    });

    req.on('error', () => {
      resolve({
        ...server,
        status: 'not running',
        statusCode: null
      });
    });

    req.on('timeout', () => {
      req.destroy();
      resolve({
        ...server,
        status: 'timeout',
        statusCode: null
      });
    });

    req.end();
  });
}

async function main() {
  console.log('\n🔍 Checking SatyaAI Servers...\n');

  const results = await Promise.all(servers.map(checkServer));

  let allRunning = true;

  results.forEach(result => {
    const icon = result.status === 'running' ? '✅' : '❌';
    const status = result.status === 'running' 
      ? `Running (${result.statusCode})` 
      : result.status === 'timeout'
      ? 'Timeout'
      : 'Not Running';
    
    console.log(`${icon} ${result.name.padEnd(20)} - ${status}`);
    
    if (result.status !== 'running') {
      allRunning = false;
    }
  });

  console.log('\n' + '='.repeat(50));

  if (allRunning) {
    console.log('✅ All servers are running!\n');
    console.log('You can now use the application at:');
    console.log('   http://localhost:5173\n');
  } else {
    console.log('❌ Some servers are not running!\n');
    console.log('To start all servers, run:');
    console.log('   npm run start:all\n');
    
    // Specific instructions for missing servers
    results.forEach(result => {
      if (result.status !== 'running') {
        if (result.name === 'Python AI Server') {
          console.log('⚠️  Python AI Server is not running!');
          console.log('   This is why you\'re getting fallback responses.');
          console.log('   Start it with: npm run dev:python\n');
        }
      }
    });
  }
}

main().catch(console.error);
