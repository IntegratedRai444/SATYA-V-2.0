import { config } from 'dotenv';
import path from 'path';

// Load test environment variables
config({
  path: path.resolve(process.cwd(), '.env.test')
});

// Mock logger to prevent test logs from cluttering the output
jest.mock('../../server/config/logger', () => ({
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
  http: jest.fn()
}));

// Mock database connection
jest.mock('../../server/db', () => ({
  db: {
    query: jest.fn(),
    transaction: jest.fn(),
    // Add other database methods as needed
  }
}));

// Global test setup
beforeAll(async () => {
  // Initialize test database or other global setup
});

afterAll(async () => {
  // Clean up test database or other global teardown
});

afterEach(() => {
  jest.clearAllMocks();
});
