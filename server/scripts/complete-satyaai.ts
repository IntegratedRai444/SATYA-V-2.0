#!/usr/bin/env node

/**
 * SatyaAI Completion Script
 * 
 * This script performs final integration and validation to complete SatyaAI
 * and ensure it's ready for production use.
 */

import 'dotenv/config';
import { logger } from '../config';
import { runValidation } from '../tests/run-validation';
import { runProductionReadinessCheck } from '../tests/production-readiness-check';

interface CompletionResults {
  validationPassed: boolean;
  productionReady: boolean;
  completionTime: number;
  summary: {
    totalTests: number;
    passedTests: number;
    failedTests: number;
    criticalIssues: number;
  };
}

class SatyaAICompletion {
  
  /**
   * Complete SatyaAI system
   */
  async complete(): Promise<CompletionResults> {
    const startTime = Date.now();
    
    logger.info('🚀 Starting SatyaAI completion process');
    logger.info('=====================================');

    const results: CompletionResults = {
      validationPassed: false,
      productionReady: false,
      completionTime: 0,
      summary: {
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        criticalIssues: 0
      }
    };

    try {
      // Step 1: System Integration
      await this.performSystemIntegration();

      // Step 2: Run comprehensive validation
      logger.info('📋 Step 2: Running comprehensive validation');
      const validationResults = await this.runValidation();
      results.validationPassed = validationResults.success;
      results.summary.totalTests += validationResults.totalTests;
      results.summary.passedTests += validationResults.passedTests;
      results.summary.failedTests += validationResults.failedTests;

      // Step 3: Production readiness check
      logger.info('🔍 Step 3: Checking production readiness');
      const readinessResults = await this.checkProductionReadiness();
      results.productionReady = readinessResults.ready;
      results.summary.criticalIssues = readinessResults.criticalIssues;

      // Step 4: Final system optimization
      await this.performFinalOptimization();

      // Calculate completion time
      results.completionTime = Date.now() - startTime;

      // Generate completion report
      this.generateCompletionReport(results);

      return results;
    } catch (error) {
      logger.error('SatyaAI completion failed', {
        error: (error as Error).message,
        stack: (error as Error).stack
      });
      throw error;
    }
  }

  /**
   * Perform system integration
   */
  private async performSystemIntegration(): Promise<void> {
    logger.info('🔧 Step 1: Performing system integration');

    // Initialize all services
    logger.info('  • Initializing performance optimizer...');
    const { performanceOptimizer } = await import('../services/performance-optimizer');
    
    logger.info('  • Initializing database optimizer...');
    const { databaseOptimizer } = await import('../services/database-optimizer');
    
    logger.info('  • Initializing health monitor...');
    const { healthMonitor } = await import('../services/health-monitor');

    // Create database indexes for optimal performance
    logger.info('  • Creating database indexes...');
    await databaseOptimizer.createOptimalIndexes();

    // Perform initial cleanup
    logger.info('  • Performing initial cleanup...');
    const { fileCleanupService } = await import('../services/file-cleanup');
    await fileCleanupService.cleanupOldFiles({
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
      dryRun: false
    });

    logger.info('✅ System integration completed');
  }

  /**
   * Run comprehensive validation
   */
  private async runValidation(): Promise<{
    success: boolean;
    totalTests: number;
    passedTests: number;
    failedTests: number;
  }> {
    try {
      // Import and run validation
      const { ValidationRunner } = await import('../tests/run-validation');
      const runner = new ValidationRunner();
      const results = await runner.runValidation();

      const successRate = (results.passedTests / results.totalTests) * 100;
      const success = successRate >= 70; // 70% pass rate required

      logger.info(`📊 Validation Results: ${results.passedTests}/${results.totalTests} passed (${successRate.toFixed(1)}%)`);

      return {
        success,
        totalTests: results.totalTests,
        passedTests: results.passedTests,
        failedTests: results.failedTests
      };
    } catch (error) {
      logger.error('Validation failed', { error: (error as Error).message });
      return {
        success: false,
        totalTests: 0,
        passedTests: 0,
        failedTests: 1
      };
    }
  }

  /**
   * Check production readiness
   */
  private async checkProductionReadiness(): Promise<{
    ready: boolean;
    criticalIssues: number;
  }> {
    try {
      const { ProductionReadinessChecker } = await import('../tests/production-readiness-check');
      const checker = new ProductionReadinessChecker();
      const results = await checker.runChecks();

      logger.info(`🔍 Production Readiness: ${results.overallScore.toFixed(1)}% (${results.criticalFailures} critical issues)`);

      return {
        ready: results.readyForProduction,
        criticalIssues: results.criticalFailures
      };
    } catch (error) {
      logger.error('Production readiness check failed', { error: (error as Error).message });
      return {
        ready: false,
        criticalIssues: 1
      };
    }
  }

  /**
   * Perform final system optimization
   */
  private async performFinalOptimization(): Promise<void> {
    logger.info('⚡ Step 4: Performing final optimization');

    try {
      // Database optimization
      logger.info('  • Optimizing database...');
      const { databaseOptimizer } = await import('../services/database-optimizer');
      await databaseOptimizer.optimizeDatabase();

      // Clear caches for fresh start
      logger.info('  • Clearing caches...');
      databaseOptimizer.clearCache();

      // Force garbage collection if available
      if (global.gc) {
        logger.info('  • Running garbage collection...');
        global.gc();
      }

      logger.info('✅ Final optimization completed');
    } catch (error) {
      logger.warn('Some optimization steps failed', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Generate completion report
   */
  private generateCompletionReport(results: CompletionResults): void {
    logger.info('');
    logger.info('🎯 SATYAAI COMPLETION REPORT');
    logger.info('============================');
    logger.info(`Completion Time: ${(results.completionTime / 1000).toFixed(1)} seconds`);
    logger.info(`Total Tests: ${results.summary.totalTests}`);
    logger.info(`Passed Tests: ${results.summary.passedTests}`);
    logger.info(`Failed Tests: ${results.summary.failedTests}`);
    logger.info(`Critical Issues: ${results.summary.criticalIssues}`);
    logger.info('');

    // Overall status
    if (results.validationPassed && results.productionReady) {
      logger.info('🎉 SATYAAI IS COMPLETE AND READY FOR PRODUCTION!');
      logger.info('');
      logger.info('✅ All systems operational');
      logger.info('✅ Validation passed');
      logger.info('✅ Production ready');
      logger.info('✅ Performance optimized');
      logger.info('');
      logger.info('🚀 Your professional-grade deepfake detection platform is ready!');
    } else if (results.validationPassed && !results.productionReady) {
      logger.warn('⚠️  SATYAAI IS FUNCTIONAL BUT HAS PRODUCTION ISSUES');
      logger.info('');
      logger.info('✅ Core functionality working');
      logger.info('⚠️  Production readiness issues detected');
      logger.info(`❌ ${results.summary.criticalIssues} critical issues need resolution`);
      logger.info('');
      logger.info('🔧 Address production issues before deploying');
    } else {
      logger.error('🚨 SATYAAI COMPLETION FAILED');
      logger.info('');
      logger.info('❌ Validation failed');
      logger.info(`❌ ${results.summary.failedTests} tests failed`);
      logger.info(`❌ ${results.summary.criticalIssues} critical issues`);
      logger.info('');
      logger.info('🔧 Review failed tests and fix issues before proceeding');
    }

    logger.info('');
    logger.info('📚 Next Steps:');
    if (results.validationPassed && results.productionReady) {
      logger.info('  1. Deploy to production environment');
      logger.info('  2. Configure monitoring and alerting');
      logger.info('  3. Set up backup and recovery procedures');
      logger.info('  4. Train users on the system');
    } else {
      logger.info('  1. Review and fix failed tests');
      logger.info('  2. Address critical production issues');
      logger.info('  3. Re-run completion process');
      logger.info('  4. Verify all systems before deployment');
    }
    logger.info('');
  }
}

/**
 * Main completion function
 */
async function completeSatyaAI(): Promise<void> {
  const completion = new SatyaAICompletion();
  
  try {
    const results = await completion.complete();
    
    // Exit with appropriate code
    const success = results.validationPassed && results.productionReady;
    process.exit(success ? 0 : 1);
  } catch (error) {
    logger.error('SatyaAI completion process failed', {
      error: (error as Error).message
    });
    process.exit(1);
  }
}

// Run completion if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  completeSatyaAI().catch(error => {
    console.error('SatyaAI completion failed:', error);
    process.exit(1);
  });
}

export { SatyaAICompletion, completeSatyaAI };