#!/usr/bin/env node

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { apiClient } from './test-utils/api-client';
import { logger } from '../config';

interface ReadinessCheck {
  category: string;
  name: string;
  description: string;
  check: () => Promise<{ passed: boolean; message: string; details?: any }>;
  critical: boolean;
}

interface ReadinessResults {
  totalChecks: number;
  passedChecks: number;
  failedChecks: number;
  criticalFailures: number;
  overallScore: number;
  readyForProduction: boolean;
  categories: Record<string, {
    passed: number;
    failed: number;
    total: number;
  }>;
  details: Array<{
    category: string;
    name: string;
    passed: boolean;
    message: string;
    critical: boolean;
    details?: any;
  }>;
}

class ProductionReadinessChecker {
  private checks: ReadinessCheck[] = [];

  constructor() {
    this.initializeChecks();
  }

  /**
   * Initialize all production readiness checks
   */
  private initializeChecks(): void {
    // Security Checks
    this.addSecurityChecks();
    
    // Performance Checks
    this.addPerformanceChecks();
    
    // Reliability Checks
    this.addReliabilityChecks();
    
    // Configuration Checks
    this.addConfigurationChecks();
    
    // Infrastructure Checks
    this.addInfrastructureChecks();
    
    // Monitoring Checks
    this.addMonitoringChecks();
  }

  /**
   * Add security-related checks
   */
  private addSecurityChecks(): void {
    this.checks.push(
      {
        category: 'Security',
        name: 'Environment Variables Security',
        description: 'Check that sensitive environment variables are properly configured',
        critical: true,
        check: async () => {
          const requiredEnvVars = [
            'JWT_SECRET',
            'SESSION_SECRET',
            'NODE_ENV'
          ];

          const missing = requiredEnvVars.filter(env => !process.env[env]);
          const weak = [];

          // Check JWT_SECRET strength
          if (process.env.JWT_SECRET && process.env.JWT_SECRET.length < 32) {
            weak.push('JWT_SECRET is too short (minimum 32 characters)');
          }

          // Check SESSION_SECRET strength
          if (process.env.SESSION_SECRET && process.env.SESSION_SECRET.length < 32) {
            weak.push('SESSION_SECRET is too short (minimum 32 characters)');
          }

          const issues = [...missing.map(env => `Missing: ${env}`), ...weak];

          return {
            passed: issues.length === 0,
            message: issues.length === 0 
              ? 'All security environment variables are properly configured'
              : `Security issues found: ${issues.join(', ')}`,
            details: { missing, weak }
          };
        }
      },
      {
        category: 'Security',
        name: 'Authentication System',
        description: 'Verify authentication system is working correctly',
        critical: true,
        check: async () => {
          try {
            // Test user registration
            const testUser = await apiClient.createTestUser('security_test');
            
            // Test login
            await apiClient.logout();
            const loginResult = await apiClient.login(testUser.username, testUser.password);
            
            // Test token validation
            const statusResult = await apiClient.getAuthStatus();
            
            // Cleanup
            await apiClient.cleanupTestUser(testUser.username);

            const passed = loginResult.success && statusResult.authenticated;

            return {
              passed,
              message: passed 
                ? 'Authentication system is working correctly'
                : 'Authentication system has issues',
              details: { loginResult, statusResult }
            };
          } catch (error) {
            return {
              passed: false,
              message: `Authentication test failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      },
      {
        category: 'Security',
        name: 'Rate Limiting',
        description: 'Check if rate limiting is properly configured',
        critical: false,
        check: async () => {
          // This is a simplified check - in production you'd test actual rate limiting
          const hasRateLimit = process.env.RATE_LIMIT_MAX_REQUESTS !== undefined;
          
          return {
            passed: hasRateLimit,
            message: hasRateLimit 
              ? 'Rate limiting is configured'
              : 'Rate limiting configuration not found',
            details: {
              maxRequests: process.env.RATE_LIMIT_MAX_REQUESTS,
              windowMs: process.env.RATE_LIMIT_WINDOW_MS
            }
          };
        }
      }
    );
  }

  /**
   * Add performance-related checks
   */
  private addPerformanceChecks(): void {
    this.checks.push(
      {
        category: 'Performance',
        name: 'Response Time',
        description: 'Check API response times are acceptable',
        critical: false,
        check: async () => {
          const startTime = Date.now();
          await apiClient.getHealth();
          const responseTime = Date.now() - startTime;

          const acceptable = responseTime < 2000; // 2 seconds

          return {
            passed: acceptable,
            message: `API response time: ${responseTime}ms ${acceptable ? '(Good)' : '(Too slow)'}`,
            details: { responseTime, threshold: 2000 }
          };
        }
      },
      {
        category: 'Performance',
        name: 'Memory Usage',
        description: 'Check memory usage is within acceptable limits',
        critical: false,
        check: async () => {
          const metrics = await apiClient.getMetrics();
          const memoryUsageMB = metrics.system?.memory?.heapUsedMB || 0;
          const acceptable = memoryUsageMB < 500; // 500MB threshold

          return {
            passed: acceptable,
            message: `Memory usage: ${memoryUsageMB}MB ${acceptable ? '(Good)' : '(High)'}`,
            details: { memoryUsageMB, threshold: 500 }
          };
        }
      },
      {
        category: 'Performance',
        name: 'File Upload Limits',
        description: 'Verify file upload limits are properly configured',
        critical: true,
        check: async () => {
          const maxFileSize = process.env.MAX_FILE_SIZE;
          const hasLimit = maxFileSize !== undefined;
          const reasonableLimit = hasLimit && parseInt(maxFileSize) <= 100 * 1024 * 1024; // 100MB

          return {
            passed: hasLimit && reasonableLimit,
            message: hasLimit 
              ? `File upload limit: ${Math.round(parseInt(maxFileSize) / 1024 / 1024)}MB ${reasonableLimit ? '(Good)' : '(Too high)'}`
              : 'File upload limit not configured',
            details: { maxFileSize, reasonableLimit }
          };
        }
      }
    );
  }

  /**
   * Add reliability-related checks
   */
  private addReliabilityChecks(): void {
    this.checks.push(
      {
        category: 'Reliability',
        name: 'Health Check Endpoints',
        description: 'Verify health check endpoints are working',
        critical: true,
        check: async () => {
          try {
            const [basicHealth, detailedHealth] = await Promise.all([
              apiClient.getHealth(),
              apiClient.getDetailedHealth()
            ]);

            const basicWorking = basicHealth.status !== undefined;
            const detailedWorking = detailedHealth.components !== undefined;

            return {
              passed: basicWorking && detailedWorking,
              message: `Health endpoints: Basic ${basicWorking ? '✓' : '✗'}, Detailed ${detailedWorking ? '✓' : '✗'}`,
              details: { basicHealth, detailedHealth }
            };
          } catch (error) {
            return {
              passed: false,
              message: `Health check failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      },
      {
        category: 'Reliability',
        name: 'Error Handling',
        description: 'Test error handling and recovery',
        critical: true,
        check: async () => {
          try {
            // Test with invalid request
            const invalidResult = await apiClient.analyzeImage(
              new File(['invalid'], 'test.txt', { type: 'text/plain' }),
              { async: false }
            );

            const hasErrorHandling = !invalidResult.success && 
                                   invalidResult.error && 
                                   typeof invalidResult.error.message === 'string';

            return {
              passed: hasErrorHandling,
              message: hasErrorHandling 
                ? 'Error handling is working correctly'
                : 'Error handling needs improvement',
              details: { invalidResult }
            };
          } catch (error) {
            // If it throws, that's also acceptable error handling
            return {
              passed: true,
              message: 'Error handling is working (throws exceptions)',
              details: { threwException: true }
            };
          }
        }
      },
      {
        category: 'Reliability',
        name: 'Database Connectivity',
        description: 'Verify database is accessible and responsive',
        critical: true,
        check: async () => {
          try {
            const health = await apiClient.getDetailedHealth();
            const dbStatus = health.components?.database?.status;
            const dbResponseTime = health.components?.database?.responseTime;

            const passed = dbStatus === 'healthy' || dbStatus === 'connected';
            const fastResponse = dbResponseTime ? dbResponseTime < 1000 : true;

            return {
              passed: passed && fastResponse,
              message: `Database: ${dbStatus}, Response time: ${dbResponseTime}ms`,
              details: { dbStatus, dbResponseTime }
            };
          } catch (error) {
            return {
              passed: false,
              message: `Database check failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      }
    );
  }

  /**
   * Add configuration-related checks
   */
  private addConfigurationChecks(): void {
    this.checks.push(
      {
        category: 'Configuration',
        name: 'Production Environment',
        description: 'Verify NODE_ENV is set to production',
        critical: true,
        check: async () => {
          const isProduction = process.env.NODE_ENV === 'production';
          
          return {
            passed: isProduction,
            message: `NODE_ENV: ${process.env.NODE_ENV} ${isProduction ? '(Correct)' : '(Should be production)'}`,
            details: { nodeEnv: process.env.NODE_ENV }
          };
        }
      },
      {
        category: 'Configuration',
        name: 'CORS Configuration',
        description: 'Check CORS is properly configured',
        critical: true,
        check: async () => {
          const corsOrigin = process.env.CORS_ORIGIN;
          const hasConfig = corsOrigin !== undefined;
          const notWildcard = corsOrigin !== '*';

          return {
            passed: hasConfig && notWildcard,
            message: hasConfig 
              ? `CORS configured: ${corsOrigin} ${notWildcard ? '(Secure)' : '(Insecure wildcard)'}`
              : 'CORS not configured',
            details: { corsOrigin, hasConfig, notWildcard }
          };
        }
      },
      {
        category: 'Configuration',
        name: 'Logging Configuration',
        description: 'Verify logging is properly configured',
        critical: false,
        check: async () => {
          // Check if logs directory exists and is writable
          try {
            const logsDir = path.join(process.cwd(), 'logs');
            await fs.access(logsDir);
            
            return {
              passed: true,
              message: 'Logging directory is accessible',
              details: { logsDir }
            };
          } catch (error) {
            return {
              passed: false,
              message: 'Logging directory not found or not accessible',
              details: { error: (error as Error).message }
            };
          }
        }
      }
    );
  }

  /**
   * Add infrastructure-related checks
   */
  private addInfrastructureChecks(): void {
    this.checks.push(
      {
        category: 'Infrastructure',
        name: 'Python AI Service',
        description: 'Verify Python AI service is running and accessible',
        critical: true,
        check: async () => {
          try {
            const health = await apiClient.getDetailedHealth();
            const pythonStatus = health.components?.python?.status;
            const pythonResponseTime = health.components?.python?.responseTime;

            const passed = pythonStatus === 'connected' || pythonStatus === 'healthy';

            return {
              passed,
              message: `Python AI service: ${pythonStatus}, Response time: ${pythonResponseTime}ms`,
              details: { pythonStatus, pythonResponseTime }
            };
          } catch (error) {
            return {
              passed: false,
              message: `Python AI service check failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      },
      {
        category: 'Infrastructure',
        name: 'WebSocket Support',
        description: 'Verify WebSocket functionality is available',
        critical: false,
        check: async () => {
          try {
            const metrics = await apiClient.getMetrics();
            const wsSupported = metrics.application?.websocket !== undefined;

            return {
              passed: wsSupported,
              message: wsSupported 
                ? 'WebSocket support is available'
                : 'WebSocket support not detected',
              details: { websocketMetrics: metrics.application?.websocket }
            };
          } catch (error) {
            return {
              passed: false,
              message: `WebSocket check failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      },
      {
        category: 'Infrastructure',
        name: 'File System Permissions',
        description: 'Check file system permissions for uploads and temp files',
        critical: true,
        check: async () => {
          try {
            const uploadDir = process.env.UPLOAD_DIR || './uploads';
            const tempDir = process.env.TEMP_DIR || './temp';

            const checks = await Promise.allSettled([
              fs.access(uploadDir, fs.constants.W_OK),
              fs.access(tempDir, fs.constants.W_OK)
            ]);

            const uploadWritable = checks[0].status === 'fulfilled';
            const tempWritable = checks[1].status === 'fulfilled';

            return {
              passed: uploadWritable && tempWritable,
              message: `Upload dir: ${uploadWritable ? '✓' : '✗'}, Temp dir: ${tempWritable ? '✓' : '✗'}`,
              details: { uploadDir, tempDir, uploadWritable, tempWritable }
            };
          } catch (error) {
            return {
              passed: false,
              message: `File system check failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      }
    );
  }

  /**
   * Add monitoring-related checks
   */
  private addMonitoringChecks(): void {
    this.checks.push(
      {
        category: 'Monitoring',
        name: 'Metrics Endpoint',
        description: 'Verify metrics endpoint is available for monitoring',
        critical: false,
        check: async () => {
          try {
            const metrics = await apiClient.getMetrics();
            const hasSystemMetrics = metrics.system !== undefined;
            const hasAppMetrics = metrics.application !== undefined;

            return {
              passed: hasSystemMetrics && hasAppMetrics,
              message: `Metrics available: System ${hasSystemMetrics ? '✓' : '✗'}, Application ${hasAppMetrics ? '✓' : '✗'}`,
              details: { hasSystemMetrics, hasAppMetrics }
            };
          } catch (error) {
            return {
              passed: false,
              message: `Metrics endpoint failed: ${(error as Error).message}`,
              details: { error: (error as Error).message }
            };
          }
        }
      },
      {
        category: 'Monitoring',
        name: 'Error Logging',
        description: 'Verify error logging is working',
        critical: false,
        check: async () => {
          // This is a simplified check - in production you'd verify actual log files
          const hasErrorLogging = process.env.LOG_LEVEL !== undefined;

          return {
            passed: hasErrorLogging,
            message: hasErrorLogging 
              ? `Error logging configured (level: ${process.env.LOG_LEVEL})`
              : 'Error logging configuration not found',
            details: { logLevel: process.env.LOG_LEVEL }
          };
        }
      }
    );
  }

  /**
   * Run all production readiness checks
   */
  async runChecks(): Promise<ReadinessResults> {
    logger.info('🔍 Starting production readiness checks');

    const results: ReadinessResults = {
      totalChecks: this.checks.length,
      passedChecks: 0,
      failedChecks: 0,
      criticalFailures: 0,
      overallScore: 0,
      readyForProduction: false,
      categories: {},
      details: []
    };

    // Initialize category counters
    for (const check of this.checks) {
      if (!results.categories[check.category]) {
        results.categories[check.category] = { passed: 0, failed: 0, total: 0 };
      }
      results.categories[check.category].total++;
    }

    // Run all checks
    for (const check of this.checks) {
      try {
        logger.info(`🔍 Running: ${check.category} - ${check.name}`);
        
        const result = await check.check();
        
        const detail = {
          category: check.category,
          name: check.name,
          passed: result.passed,
          message: result.message,
          critical: check.critical,
          details: result.details
        };

        results.details.push(detail);

        if (result.passed) {
          results.passedChecks++;
          results.categories[check.category].passed++;
          logger.info(`✅ ${check.category} - ${check.name}: ${result.message}`);
        } else {
          results.failedChecks++;
          results.categories[check.category].failed++;
          
          if (check.critical) {
            results.criticalFailures++;
            logger.error(`🚨 CRITICAL - ${check.category} - ${check.name}: ${result.message}`);
          } else {
            logger.warn(`⚠️  ${check.category} - ${check.name}: ${result.message}`);
          }
        }
      } catch (error) {
        results.failedChecks++;
        results.categories[check.category].failed++;
        
        if (check.critical) {
          results.criticalFailures++;
        }

        const detail = {
          category: check.category,
          name: check.name,
          passed: false,
          message: `Check failed: ${(error as Error).message}`,
          critical: check.critical,
          details: { error: (error as Error).message }
        };

        results.details.push(detail);
        logger.error(`💥 ${check.category} - ${check.name}: Check failed - ${(error as Error).message}`);
      }
    }

    // Calculate overall score and readiness
    results.overallScore = (results.passedChecks / results.totalChecks) * 100;
    results.readyForProduction = results.criticalFailures === 0 && results.overallScore >= 80;

    this.generateReport(results);

    return results;
  }

  /**
   * Generate production readiness report
   */
  private generateReport(results: ReadinessResults): void {
    logger.info('📊 Production Readiness Report');
    logger.info(`Overall Score: ${results.overallScore.toFixed(1)}%`);
    logger.info(`Passed: ${results.passedChecks}/${results.totalChecks}`);
    logger.info(`Critical Failures: ${results.criticalFailures}`);

    // Category breakdown
    logger.info('📋 Category Breakdown:');
    for (const [category, stats] of Object.entries(results.categories)) {
      const categoryScore = (stats.passed / stats.total) * 100;
      logger.info(`  ${category}: ${stats.passed}/${stats.total} (${categoryScore.toFixed(1)}%)`);
    }

    // Production readiness verdict
    if (results.readyForProduction) {
      logger.info('🎉 READY FOR PRODUCTION!');
    } else if (results.criticalFailures > 0) {
      logger.error(`🚨 NOT READY - ${results.criticalFailures} critical issues must be resolved`);
    } else {
      logger.warn('⚠️  PARTIALLY READY - Some non-critical issues should be addressed');
    }
  }
}

/**
 * Main function to run production readiness check
 */
async function runProductionReadinessCheck(): Promise<void> {
  const checker = new ProductionReadinessChecker();
  
  try {
    const results = await checker.runChecks();
    
    // Exit with appropriate code
    process.exit(results.readyForProduction ? 0 : 1);
  } catch (error) {
    logger.error('Production readiness check failed', {
      error: (error as Error).message
    });
    process.exit(1);
  }
}

// Run check if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runProductionReadinessCheck().catch(error => {
    console.error('Production readiness check failed:', error);
    process.exit(1);
  });
}

export { ProductionReadinessChecker, runProductionReadinessCheck };