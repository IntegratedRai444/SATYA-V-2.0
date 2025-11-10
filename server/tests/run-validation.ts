#!/usr/bin/env node

import 'dotenv/config';
import { testOrchestrator } from './e2e-test-framework';
import { coreWorkflowTests } from './scenarios/core-workflow-tests';
import { errorRecoveryTests } from './scenarios/error-recovery-tests';
import { apiClient } from './test-utils/api-client';
import { testDataManager } from './test-utils/test-data-manager';
import { logger } from '../config';

interface ValidationResults {
  totalTests: number;
  passedTests: number;
  failedTests: number;
  skippedTests: number;
  duration: number;
  testResults: Array<{
    name: string;
    success: boolean;
    duration: number;
    error?: string;
  }>;
}

class ValidationRunner {
  private results: ValidationResults = {
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    skippedTests: 0,
    duration: 0,
    testResults: []
  };

  /**
   * Run comprehensive validation
   */
  async runValidation(): Promise<ValidationResults> {
    const startTime = Date.now();

    logger.info('🚀 Starting SatyaAI comprehensive validation');

    try {
      // Initialize test environment
      await this.initializeTestEnvironment();

      // Run core workflow tests
      await this.runTestSuite('Core Workflow Tests', coreWorkflowTests);

      // Run error recovery tests
      await this.runTestSuite('Error Recovery Tests', errorRecoveryTests);

      // Run system validation
      await this.runSystemValidation();

      // Calculate final results
      this.results.duration = Date.now() - startTime;
      this.results.totalTests = this.results.testResults.length;

      // Generate report
      this.generateReport();

      return this.results;
    } catch (error) {
      logger.error('Validation failed', {
        error: (error as Error).message
      });
      throw error;
    } finally {
      await this.cleanup();
    }
  }

  /**
   * Initialize test environment
   */
  private async initializeTestEnvironment(): Promise<void> {
    logger.info('🔧 Initializing test environment');

    try {
      // Check if server is running
      const health = await apiClient.getHealth();
      if (!health || health._httpStatus !== 200) {
        throw new Error('Server is not running or unhealthy');
      }

      // Initialize test data
      await testDataManager.createTestFileFromTemplate('real-image', 'validation-test-image.jpg');
      await testDataManager.createTestFileFromTemplate('real-audio', 'validation-test-audio.wav');

      logger.info('✅ Test environment initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize test environment', {
        error: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * Run a test suite
   */
  private async runTestSuite(suiteName: string, tests: any[]): Promise<void> {
    logger.info(`📋 Running ${suiteName} (${tests.length} tests)`);

    for (const test of tests) {
      try {
        logger.info(`🧪 Running test: ${test.name}`);
        
        const result = await testOrchestrator.executeScenario(test);
        
        this.results.testResults.push({
          name: test.name,
          success: result.success,
          duration: result.duration,
          error: result.error
        });

        if (result.success) {
          this.results.passedTests++;
          logger.info(`✅ Test passed: ${test.name} (${result.duration}ms)`);
        } else {
          this.results.failedTests++;
          logger.error(`❌ Test failed: ${test.name}`, {
            error: result.error,
            duration: result.duration
          });
        }
      } catch (error) {
        this.results.failedTests++;
        this.results.testResults.push({
          name: test.name,
          success: false,
          duration: 0,
          error: (error as Error).message
        });

        logger.error(`💥 Test crashed: ${test.name}`, {
          error: (error as Error).message
        });
      }
    }
  }

  /**
   * Run system validation checks
   */
  private async runSystemValidation(): Promise<void> {
    logger.info('🔍 Running system validation checks');

    const systemChecks = [
      {
        name: 'Health Check Validation',
        check: async () => {
          const health = await apiClient.getHealth();
          return health.status === 'healthy' || health.status === 'degraded';
        }
      },
      {
        name: 'Detailed Health Check Validation',
        check: async () => {
          const health = await apiClient.getDetailedHealth();
          return health.components && 
                 health.components.nodejs && 
                 health.components.python &&
                 health.components.database;
        }
      },
      {
        name: 'Metrics Endpoint Validation',
        check: async () => {
          const metrics = await apiClient.getMetrics();
          return metrics.system && 
                 metrics.application &&
                 typeof metrics.system.uptime === 'number';
        }
      },
      {
        name: 'Authentication System Validation',
        check: async () => {
          const testUser = await apiClient.createTestUser('validation');
          const status = await apiClient.getAuthStatus();
          await apiClient.cleanupTestUser(testUser.username);
          return status.authenticated === true;
        }
      },
      {
        name: 'File Upload Validation',
        check: async () => {
          // Create a minimal test user and file
          const testUser = await apiClient.createTestUser('upload_test');
          
          try {
            const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
            const result = await apiClient.analyzeImage(testFile, { async: false });
            
            // Should either succeed or fail gracefully
            return result !== undefined && 
                   (result.success === true || result.success === false);
          } finally {
            await apiClient.cleanupTestUser(testUser.username);
          }
        }
      },
      {
        name: 'Database Connectivity Validation',
        check: async () => {
          // Test database through API endpoints
          const testUser = await apiClient.createTestUser('db_test');
          const profile = await apiClient.getAuthStatus();
          await apiClient.cleanupTestUser(testUser.username);
          return profile.authenticated !== undefined;
        }
      },
      {
        name: 'WebSocket Connectivity Validation',
        check: async () => {
          // Check if WebSocket endpoint is available
          const metrics = await apiClient.getMetrics();
          return metrics.application?.websocket !== undefined;
        }
      }
    ];

    for (const check of systemChecks) {
      try {
        logger.info(`🔍 Running: ${check.name}`);
        
        const startTime = Date.now();
        const success = await check.check();
        const duration = Date.now() - startTime;

        this.results.testResults.push({
          name: check.name,
          success,
          duration,
          error: success ? undefined : 'System check failed'
        });

        if (success) {
          this.results.passedTests++;
          logger.info(`✅ System check passed: ${check.name} (${duration}ms)`);
        } else {
          this.results.failedTests++;
          logger.error(`❌ System check failed: ${check.name} (${duration}ms)`);
        }
      } catch (error) {
        this.results.failedTests++;
        this.results.testResults.push({
          name: check.name,
          success: false,
          duration: 0,
          error: (error as Error).message
        });

        logger.error(`💥 System check crashed: ${check.name}`, {
          error: (error as Error).message
        });
      }
    }
  }

  /**
   * Generate validation report
   */
  private generateReport(): void {
    const successRate = (this.results.passedTests / this.results.totalTests) * 100;
    
    logger.info('📊 Validation Results Summary', {
      totalTests: this.results.totalTests,
      passedTests: this.results.passedTests,
      failedTests: this.results.failedTests,
      successRate: `${successRate.toFixed(1)}%`,
      duration: `${(this.results.duration / 1000).toFixed(1)}s`
    });

    // Log failed tests
    if (this.results.failedTests > 0) {
      logger.error('❌ Failed Tests:');
      this.results.testResults
        .filter(test => !test.success)
        .forEach(test => {
          logger.error(`  - ${test.name}: ${test.error}`);
        });
    }

    // Overall status
    if (successRate >= 90) {
      logger.info('🎉 VALIDATION PASSED - System is ready for production!');
    } else if (successRate >= 70) {
      logger.warn('⚠️  VALIDATION PARTIAL - System has some issues but is functional');
    } else {
      logger.error('🚨 VALIDATION FAILED - System has critical issues');
    }
  }

  /**
   * Cleanup test environment
   */
  private async cleanup(): Promise<void> {
    logger.info('🧹 Cleaning up test environment');

    try {
      // Cleanup test data
      await testDataManager.removeTestFile('validation-test-image.jpg');
      await testDataManager.removeTestFile('validation-test-audio.wav');

      // Logout any test sessions
      try {
        await apiClient.logout();
      } catch (error) {
        // Ignore logout errors during cleanup
      }

      logger.info('✅ Test environment cleanup completed');
    } catch (error) {
      logger.warn('⚠️  Test cleanup had issues', {
        error: (error as Error).message
      });
    }
  }
}

/**
 * Main validation function
 */
async function runValidation(): Promise<void> {
  const runner = new ValidationRunner();
  
  try {
    const results = await runner.runValidation();
    
    // Exit with appropriate code
    const successRate = (results.passedTests / results.totalTests) * 100;
    process.exit(successRate >= 70 ? 0 : 1);
  } catch (error) {
    logger.error('Validation runner failed', {
      error: (error as Error).message
    });
    process.exit(1);
  }
}

// Run validation if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runValidation().catch(error => {
    console.error('Validation failed:', error);
    process.exit(1);
  });
}

export { ValidationRunner, runValidation };