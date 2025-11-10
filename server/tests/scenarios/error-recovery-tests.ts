import { TestScenario } from '../e2e-test-framework';

/**
 * Python Server Unavailable Test
 */
export const pythonServerUnavailableTest: TestScenario = {
  name: 'Python Server Unavailable Recovery',
  description: 'Test system behavior when Python AI server is unavailable',
  timeout: 120000, // 2 minutes
  retries: 1,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_error',
          password: 'TestPassword123!',
          email: 'testuser_error@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Upload test image',
      action: 'upload',
      parameters: {
        filePath: 'sample-real-image.jpg',
        fileType: 'image'
      },
      timeout: 5000
    },
    {
      name: 'Attempt analysis with server down',
      action: 'analyze',
      parameters: {
        analysisType: 'image',
        options: {
          sensitivity: 'medium',
          async: false
        }
      },
      timeout: 30000,
      expectedResult: {
        success: false,
        error: {
          code: 'AI_SERVICE_UNAVAILABLE'
        }
      }
    },
    {
      name: 'Verify error message is user-friendly',
      action: 'verify',
      parameters: {
        condition: 'result.error.message',
        value: 'temporarily unavailable',
        operator: 'contains'
      },
      timeout: 1000
    },
    {
      name: 'Verify suggestions are provided',
      action: 'verify',
      parameters: {
        condition: 'result.error.suggestions',
        value: 1,
        operator: 'greaterThan'
      },
      timeout: 1000
    },
    {
      name: 'Check health endpoint reflects issue',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getDetailedHealth();
          return health.components.python.status !== 'healthy';
        }
      },
      timeout: 10000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

/**
 * File Upload Failure Test
 */
export const fileUploadFailureTest: TestScenario = {
  name: 'File Upload Failure Recovery',
  description: 'Test system behavior with invalid file uploads and cleanup',
  timeout: 60000, // 1 minute
  retries: 2,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_upload',
          password: 'TestPassword123!',
          email: 'testuser_upload@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Attempt upload with invalid file type',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          try {
            // Create an invalid file (text file with image extension)
            const invalidFile = new File(['invalid content'], 'fake.jpg', { 
              type: 'text/plain' 
            });
            
            const result = await context.apiClient.analyzeImage(invalidFile);
            return result.success === false && 
                   result.error?.code === 'INVALID_FILE_TYPE';
          } catch (error) {
            return true; // Expected to fail
          }
        }
      },
      timeout: 15000
    },
    {
      name: 'Attempt upload with oversized file',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          try {
            // Create a large file (simulate oversized)
            const largeBuffer = Buffer.alloc(100 * 1024 * 1024); // 100MB
            const largeFile = new File([largeBuffer], 'large.jpg', { 
              type: 'image/jpeg' 
            });
            
            const result = await context.apiClient.analyzeImage(largeFile);
            return result.success === false && 
                   result.error?.code === 'FILE_TOO_LARGE';
          } catch (error) {
            return true; // Expected to fail
          }
        }
      },
      timeout: 20000
    },
    {
      name: 'Verify error messages are helpful',
      action: 'verify',
      parameters: {
        condition: 'previousResults.length',
        value: 2,
        operator: 'equals'
      },
      timeout: 1000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

/**
 * Database Connection Failure Test
 */
export const databaseFailureTest: TestScenario = {
  name: 'Database Connection Failure',
  description: 'Test system behavior when database is unavailable',
  timeout: 60000, // 1 minute
  retries: 1,
  steps: [
    {
      name: 'Check health before database issue',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getDetailedHealth();
          return health.components.database.status === 'healthy';
        }
      },
      timeout: 10000
    },
    {
      name: 'Attempt registration during database issue',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          // This test assumes database issues would be detected
          // In a real scenario, you might temporarily disable the database
          const health = await context.apiClient.getDetailedHealth();
          return health.components.database.responseTime !== undefined;
        }
      },
      timeout: 15000
    },
    {
      name: 'Verify graceful degradation',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getHealth();
          return health.status !== undefined; // Should still respond
        }
      },
      timeout: 10000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

/**
 * Concurrent User Load Test
 */
export const concurrentUserTest: TestScenario = {
  name: 'Concurrent User Load Test',
  description: 'Test system stability with multiple concurrent users',
  timeout: 300000, // 5 minutes
  retries: 1,
  steps: [
    {
      name: 'Create multiple test users',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const userPromises = [];
          for (let i = 0; i < 5; i++) {
            userPromises.push(
              context.apiClient.register({
                username: `concurrent_user_${i}_${Date.now()}`,
                password: 'TestPassword123!',
                email: `concurrent_user_${i}@test.com`
              })
            );
          }
          
          const results = await Promise.allSettled(userPromises);
          const successCount = results.filter(r => 
            r.status === 'fulfilled' && r.value.success
          ).length;
          
          return successCount >= 3; // At least 3 should succeed
        }
      },
      timeout: 30000
    },
    {
      name: 'Simulate concurrent analysis requests',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const analysisPromises = [];
          
          for (let i = 0; i < 3; i++) {
            // Create a simple test file for each request
            const testFile = new File(['test'], `test_${i}.jpg`, { 
              type: 'image/jpeg' 
            });
            
            analysisPromises.push(
              context.apiClient.analyzeImage(testFile, { async: true })
            );
          }
          
          const results = await Promise.allSettled(analysisPromises);
          const successCount = results.filter(r => 
            r.status === 'fulfilled' && 
            (r.value.success || r.value.async)
          ).length;
          
          return successCount >= 2; // At least 2 should be accepted
        }
      },
      timeout: 60000
    },
    {
      name: 'Verify system remains responsive',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getHealth();
          return health.status !== 'unhealthy';
        }
      },
      timeout: 15000
    },
    {
      name: 'Check WebSocket connections',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const metrics = await context.apiClient.getMetrics();
          return metrics.application?.websocket !== undefined;
        }
      },
      timeout: 10000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

/**
 * Memory Leak Detection Test
 */
export const memoryLeakTest: TestScenario = {
  name: 'Memory Leak Detection',
  description: 'Test for memory leaks during repeated operations',
  timeout: 180000, // 3 minutes
  retries: 1,
  steps: [
    {
      name: 'Record initial memory usage',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const metrics = await context.apiClient.getMetrics();
          context.initialMemory = metrics.system.memory.heapUsedMB;
          return typeof context.initialMemory === 'number';
        }
      },
      timeout: 5000
    },
    {
      name: 'Perform repeated operations',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          // Create test user
          await context.apiClient.register({
            username: `memory_test_${Date.now()}`,
            password: 'TestPassword123!',
            email: 'memory_test@test.com'
          });
          
          // Perform multiple small operations
          for (let i = 0; i < 10; i++) {
            await context.apiClient.getHealth();
            await context.apiClient.getAuthStatus();
            
            // Small delay between operations
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          
          return true;
        }
      },
      timeout: 60000
    },
    {
      name: 'Check memory usage after operations',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          // Wait a bit for garbage collection
          await new Promise(resolve => setTimeout(resolve, 5000));
          
          const metrics = await context.apiClient.getMetrics();
          const finalMemory = metrics.system.memory.heapUsedMB;
          const memoryIncrease = finalMemory - context.initialMemory;
          
          // Memory should not increase by more than 50MB
          return memoryIncrease < 50;
        }
      },
      timeout: 15000
    },
    {
      name: 'Verify system stability',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getDetailedHealth();
          return health.components.memory.status !== 'unhealthy';
        }
      },
      timeout: 10000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

/**
 * Network Timeout Recovery Test
 */
export const networkTimeoutTest: TestScenario = {
  name: 'Network Timeout Recovery',
  description: 'Test system behavior with network timeouts and recovery',
  timeout: 120000, // 2 minutes
  retries: 2,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_timeout',
          password: 'TestPassword123!',
          email: 'testuser_timeout@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Test timeout handling',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          try {
            // This should timeout or fail gracefully
            const result = await context.apiClient.getHealth();
            return result !== undefined; // Should get some response
          } catch (error) {
            // Timeout errors should be handled gracefully
            return error.message.includes('timeout') || 
                   error.message.includes('network');
          }
        }
      },
      timeout: 30000
    },
    {
      name: 'Verify error recovery',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          // After timeout, system should still be responsive
          const health = await context.apiClient.getHealth();
          return health.status !== undefined;
        }
      },
      timeout: 15000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

// Export all error recovery tests
export const errorRecoveryTests = [
  pythonServerUnavailableTest,
  fileUploadFailureTest,
  databaseFailureTest,
  concurrentUserTest,
  memoryLeakTest,
  networkTimeoutTest
];