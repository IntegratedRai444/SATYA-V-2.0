import { teardownTestDatabase } from './test-utils/test-database';
import { stopTestServer } from './test-utils/test-server';

export default async function globalTeardown() {
  console.log('Global Teardown: Cleaning up test environment...');
  
  // Stop the test server
  await stopTestServer();
  
  // Clean up test database if needed
  // Note: The test database is typically dropped in the test setup
  
  console.log('Global Teardown: Test environment cleaned up');
}
