import { TestScenario } from '../e2e-test-framework';
import { testDataManager } from '../test-utils/test-data-manager';

/**
 * Image Analysis End-to-End Test
 */
export const imageAnalysisTest: TestScenario = {
  name: 'Image Analysis E2E',
  description: 'Complete image analysis workflow from upload to results',
  timeout: 120000, // 2 minutes
  retries: 2,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_image',
          password: 'TestPassword123!',
          email: 'testuser_image@test.com'
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
      name: 'Analyze image synchronously',
      action: 'analyze',
      parameters: {
        analysisType: 'image',
        options: {
          sensitivity: 'medium',
          includeDetails: true,
          async: false
        }
      },
      timeout: 60000,
      expectedResult: {
        success: true,
        data: {
          results: {
            authenticity: 'real'
          }
        }
      }
    },
    {
      name: 'Verify analysis result',
      action: 'verify',
      parameters: {
        condition: 'result.data.results.confidence',
        value: 0.5,
        operator: 'greaterThan'
      },
      timeout: 1000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: [
    {
      type: 'deleteFile',
      parameters: {
        filePath: 'sample-real-image.jpg'
      }
    }
  ]
};

/**
 * Video Analysis End-to-End Test
 */
export const videoAnalysisTest: TestScenario = {
  name: 'Video Analysis E2E',
  description: 'Complete video analysis workflow with async processing',
  timeout: 300000, // 5 minutes
  retries: 1,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_video',
          password: 'TestPassword123!',
          email: 'testuser_video@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Create test video file',
      action: 'upload',
      parameters: {
        filePath: 'test-video.mp4',
        fileType: 'video'
      },
      timeout: 5000,
      onSuccess: async () => {
        // Create a test video file
        await testDataManager.createTestFileFromTemplate('real-video', 'test-video.mp4');
      }
    },
    {
      name: 'Start video analysis',
      action: 'analyze',
      parameters: {
        analysisType: 'video',
        options: {
          sensitivity: 'high',
          includeDetails: true,
          async: true
        }
      },
      timeout: 30000,
      expectedResult: {
        success: true,
        async: true,
        jobId: 'string'
      }
    },
    {
      name: 'Wait for analysis completion',
      action: 'wait',
      parameters: {
        condition: async (context: any) => {
          const jobId = context.previousResults?.jobId;
          if (!jobId) return false;
          
          const status = await context.apiClient.getAnalysisStatus(jobId);
          return status.data?.status === 'completed';
        }
      },
      timeout: 180000 // 3 minutes
    },
    {
      name: 'Get analysis results',
      action: 'verify',
      parameters: {
        condition: 'result.data.status',
        value: 'completed',
        operator: 'equals'
      },
      timeout: 5000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: [
    {
      type: 'deleteFile',
      parameters: {
        filePath: 'test-video.mp4'
      }
    }
  ]
};

/**
 * Audio Analysis End-to-End Test
 */
export const audioAnalysisTest: TestScenario = {
  name: 'Audio Analysis E2E',
  description: 'Complete audio analysis workflow',
  timeout: 180000, // 3 minutes
  retries: 2,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_audio',
          password: 'TestPassword123!',
          email: 'testuser_audio@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Upload test audio',
      action: 'upload',
      parameters: {
        filePath: 'sample-real-audio.wav',
        fileType: 'audio'
      },
      timeout: 5000
    },
    {
      name: 'Analyze audio',
      action: 'analyze',
      parameters: {
        analysisType: 'audio',
        options: {
          sensitivity: 'medium',
          includeDetails: true,
          async: false
        }
      },
      timeout: 90000,
      expectedResult: {
        success: true,
        data: {
          results: {
            authenticity: 'real'
          }
        }
      }
    },
    {
      name: 'Verify audio analysis confidence',
      action: 'verify',
      parameters: {
        condition: 'result.data.results.confidence',
        value: 0.3,
        operator: 'greaterThan'
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
 * Multimodal Analysis End-to-End Test
 */
export const multimodalAnalysisTest: TestScenario = {
  name: 'Multimodal Analysis E2E',
  description: 'Complete multimodal analysis with multiple file types',
  timeout: 400000, // 6+ minutes
  retries: 1,
  steps: [
    {
      name: 'Create test user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_multimodal',
          password: 'TestPassword123!',
          email: 'testuser_multimodal@test.com'
        }
      },
      timeout: 10000
    },
    {
      name: 'Prepare multimodal files',
      action: 'upload',
      parameters: {
        files: {
          image: 'sample-real-image.jpg',
          audio: 'sample-real-audio.wav'
        }
      },
      timeout: 10000
    },
    {
      name: 'Start multimodal analysis',
      action: 'analyze',
      parameters: {
        analysisType: 'multimodal',
        options: {
          sensitivity: 'high',
          includeDetails: true,
          async: true
        }
      },
      timeout: 60000,
      expectedResult: {
        success: true,
        async: true
      }
    },
    {
      name: 'Wait for multimodal completion',
      action: 'wait',
      parameters: {
        condition: async (context: any) => {
          const jobId = context.previousResults?.jobId;
          if (!jobId) return false;
          
          const status = await context.apiClient.getAnalysisStatus(jobId);
          return status.data?.status === 'completed' || status.data?.status === 'failed';
        }
      },
      timeout: 300000 // 5 minutes
    },
    {
      name: 'Verify multimodal results',
      action: 'verify',
      parameters: {
        condition: 'result.success',
        value: true,
        operator: 'equals'
      },
      timeout: 5000
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
 * Authentication Flow Test
 */
export const authenticationFlowTest: TestScenario = {
  name: 'Authentication Flow E2E',
  description: 'Complete authentication workflow including registration, login, and session management',
  timeout: 60000, // 1 minute
  retries: 3,
  steps: [
    {
      name: 'Register new user',
      action: 'auth',
      parameters: {
        action: 'register',
        credentials: {
          username: 'testuser_auth',
          password: 'TestPassword123!',
          email: 'testuser_auth@test.com',
          fullName: 'Test User Auth'
        }
      },
      timeout: 10000,
      expectedResult: {
        success: true,
        token: 'string'
      }
    },
    {
      name: 'Verify auth status',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const status = await context.apiClient.getAuthStatus();
          return status.authenticated === true;
        }
      },
      timeout: 5000
    },
    {
      name: 'Logout user',
      action: 'auth',
      parameters: {
        action: 'logout'
      },
      timeout: 5000,
      expectedResult: {
        success: true
      }
    },
    {
      name: 'Verify logout',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const status = await context.apiClient.getAuthStatus();
          return status.authenticated === false;
        }
      },
      timeout: 5000
    },
    {
      name: 'Login with same credentials',
      action: 'auth',
      parameters: {
        action: 'login',
        credentials: {
          username: 'testuser_auth',
          password: 'TestPassword123!'
        }
      },
      timeout: 10000,
      expectedResult: {
        success: true,
        token: 'string'
      }
    },
    {
      name: 'Refresh token',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const refreshResult = await context.apiClient.refreshToken();
          return refreshResult.success === true;
        }
      },
      timeout: 5000
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
 * System Health Check Test
 */
export const systemHealthTest: TestScenario = {
  name: 'System Health Check E2E',
  description: 'Verify all system components are healthy',
  timeout: 30000, // 30 seconds
  retries: 2,
  steps: [
    {
      name: 'Check basic health',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getHealth();
          return health.status === 'healthy' || health.status === 'degraded';
        }
      },
      timeout: 10000
    },
    {
      name: 'Check detailed health',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const health = await context.apiClient.getDetailedHealth();
          return health.components && 
                 health.components.nodejs && 
                 health.components.python &&
                 health.components.database;
        }
      },
      timeout: 15000
    },
    {
      name: 'Check system metrics',
      action: 'verify',
      parameters: {
        condition: async (context: any) => {
          const metrics = await context.apiClient.getMetrics();
          return metrics.system && 
                 metrics.application &&
                 typeof metrics.system.uptime === 'number';
        }
      },
      timeout: 5000
    }
  ],
  expectedOutcome: {
    success: true,
    duration: 0,
    steps: []
  },
  cleanup: []
};

// Export all test scenarios
export const coreWorkflowTests = [
  imageAnalysisTest,
  videoAnalysisTest,
  audioAnalysisTest,
  multimodalAnalysisTest,
  authenticationFlowTest,
  systemHealthTest
];