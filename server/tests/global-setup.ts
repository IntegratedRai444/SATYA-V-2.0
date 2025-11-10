import { setupTestDatabase } from './test-utils/test-database';
import { startTestServer, stopTestServer } from './test-utils/test-server';

export default async function globalSetup() {
  console.log('Global Setup: Starting test environment...');
  
  // Initialize test database
  await setupTestDatabase();
  
  // Start test server
  await startTestServer();
  
  console.log('Global Setup: Test environment ready');
}

// Handle any cleanup when tests are done
process.on('SIGINT', async () => {
  await stopTestServer();
  process.exit(0);
});
