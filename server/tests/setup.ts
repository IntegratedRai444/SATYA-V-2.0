// Configure test environment variables
process.env.NODE_ENV = 'test';
process.env.PORT = '0'; // Let the OS assign a random port
process.env.JWT_SECRET = 'test-secret-key';
process.env.API_KEY = 'test-api-key';

// Set test database URL if not already set
if (!process.env.TEST_DATABASE_URL) {
  process.env.TEST_DATABASE_URL = 'postgres://postgres:postgres@localhost:5432/test_satyaai';
}

// Mock external services
jest.mock('@anthropic-ai/sdk', () => ({
  Anthropic: jest.fn().mockImplementation(() => ({
    messages: {
      create: jest.fn().mockResolvedValue({
        content: [{ text: 'Mocked AI response' }]
      })
    }
  }))
}));

// Global test timeout
jest.setTimeout(30000); // 30 seconds

// Global test hooks
beforeAll(async () => {
  // Any setup that needs to happen before all tests
});

afterAll(async () => {
  // Any cleanup after all tests
});

// Test utilities
global.mockRequest = (options = {}) => {
  return {
    headers: {},
    body: {},
    params: {},
    query: {},
    user: { id: 'test-user-id', role: 'user' },
    ...options
  };
};

global.mockResponse = () => {
  const res: any = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  res.send = jest.fn().mockReturnValue(res);
  return res;
};
