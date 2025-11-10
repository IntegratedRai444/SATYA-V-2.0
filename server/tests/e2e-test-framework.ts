import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';
import { logger } from '../config';
import { apiClient } from './test-utils/api-client';
import { testDataManager } from './test-utils/test-data-manager';

export interface TestStep {
  name: string;
  action: 'upload' | 'analyze' | 'verify' | 'wait' | 'cleanup' | 'auth';
  parameters: Record<string, any>;
  timeout: number;
  expectedResult?: any;
  onSuccess?: (result: any) => void;
  onError?: (error: Error) => void;
}

export interface TestScenario {
  name: string;
  description: string;
  steps: TestStep[];
  expectedOutcome: TestResult;
  cleanup: CleanupAction[];
  timeout: number;
  retries: number;
}

export interface TestResult {
  success: boolean;
  duration: number;
  steps: StepResult[];
  error?: string;
  metadata?: Record<string, any>;
}

export interface StepResult {
  stepName: string;
  success: boolean;
  duration: number;
  result?: any;
  error?: string;
}

export interface CleanupAction {
  type: 'deleteFile' | 'clearDatabase' | 'resetState';
  parameters: Record<string, any>;
}

class TestOrchestrator extends EventEmitter {
  private runningTests: Map<string, TestExecution> = new Map();
  private testResults: Map<string, TestResult> = new Map();
  private globalTimeout = 300000; // 5 minutes default

  constructor() {
    super();
    logger.info('Test orchestrator initialized');
  }

  /**
   * Execute a test scenario
   */
  async executeScenario(scenario: TestScenario): Promise<TestResult> {
    const executionId = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    logger.info('Starting test scenario', {
      executionId,
      scenarioName: scenario.name,
      steps: scenario.steps.length
    });

    const execution: TestExecution = {
      id: executionId,
      scenario,
      startTime: new Date(),
      status: 'running',
      currentStep: 0,
      stepResults: []
    };

    this.runningTests.set(executionId, execution);
    this.emit('testStarted', execution);

    try {
      const result = await this.runScenario(execution);
      
      this.testResults.set(executionId, result);
      this.runningTests.delete(executionId);
      
      this.emit('testCompleted', { executionId, result });
      
      logger.info('Test scenario completed', {
        executionId,
        scenarioName: scenario.name,
        success: result.success,
        duration: result.duration
      });

      return result;
    } catch (error) {
      const result: TestResult = {
        success: false,
        duration: Date.now() - startTime,
        steps: execution.stepResults,
        error: (error as Error).message
      };

      this.testResults.set(executionId, result);
      this.runningTests.delete(executionId);
      
      this.emit('testFailed', { executionId, result, error });
      
      logger.error('Test scenario failed', {
        executionId,
        scenarioName: scenario.name,
        error: (error as Error).message,
        duration: result.duration
      });

      return result;
    } finally {
      // Always run cleanup
      await this.runCleanup(scenario.cleanup);
    }
  }

  /**
   * Run a test scenario
   */
  private async runScenario(execution: TestExecution): Promise<TestResult> {
    const { scenario } = execution;
    const startTime = Date.now();

    for (let i = 0; i < scenario.steps.length; i++) {
      const step = scenario.steps[i];
      execution.currentStep = i;

      logger.debug('Executing test step', {
        executionId: execution.id,
        stepIndex: i,
        stepName: step.name,
        action: step.action
      });

      const stepResult = await this.executeStep(step, execution);
      execution.stepResults.push(stepResult);

      this.emit('stepCompleted', { 
        executionId: execution.id, 
        stepIndex: i, 
        stepResult 
      });

      if (!stepResult.success) {
        throw new Error(`Step "${step.name}" failed: ${stepResult.error}`);
      }
    }

    const duration = Date.now() - startTime;
    
    return {
      success: true,
      duration,
      steps: execution.stepResults,
      metadata: {
        scenarioName: scenario.name,
        totalSteps: scenario.steps.length
      }
    };
  }

  /**
   * Execute a single test step
   */
  private async executeStep(step: TestStep, execution: TestExecution): Promise<StepResult> {
    const startTime = Date.now();

    try {
      let result: any;

      switch (step.action) {
        case 'auth':
          result = await this.executeAuthStep(step);
          break;
        case 'upload':
          result = await this.executeUploadStep(step);
          break;
        case 'analyze':
          result = await this.executeAnalyzeStep(step);
          break;
        case 'verify':
          result = await this.executeVerifyStep(step);
          break;
        case 'wait':
          result = await this.executeWaitStep(step);
          break;
        case 'cleanup':
          result = await this.executeCleanupStep(step);
          break;
        default:
          throw new Error(`Unknown step action: ${step.action}`);
      }

      // Verify expected result if provided
      if (step.expectedResult) {
        this.verifyResult(result, step.expectedResult);
      }

      const duration = Date.now() - startTime;

      if (step.onSuccess) {
        step.onSuccess(result);
      }

      return {
        stepName: step.name,
        success: true,
        duration,
        result
      };
    } catch (error) {
      const duration = Date.now() - startTime;

      if (step.onError) {
        step.onError(error as Error);
      }

      return {
        stepName: step.name,
        success: false,
        duration,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute authentication step
   */
  private async executeAuthStep(step: TestStep): Promise<any> {
    const { action, credentials } = step.parameters;

    switch (action) {
      case 'login':
        return await apiClient.login(credentials.username, credentials.password);
      case 'register':
        return await apiClient.register(credentials);
      case 'logout':
        return await apiClient.logout();
      default:
        throw new Error(`Unknown auth action: ${action}`);
    }
  }

  /**
   * Execute file upload step
   */
  private async executeUploadStep(step: TestStep): Promise<any> {
    const { filePath, fileType } = step.parameters;
    
    const fileBuffer = await testDataManager.getTestFile(filePath);
    const fileName = path.basename(filePath);
    
    // Create a File-like object for testing
    const file = new File([fileBuffer], fileName, { 
      type: this.getMimeType(fileType) 
    });

    return { file, fileName, size: fileBuffer.length };
  }

  /**
   * Execute analysis step
   */
  private async executeAnalyzeStep(step: TestStep): Promise<any> {
    const { file, analysisType, options } = step.parameters;

    switch (analysisType) {
      case 'image':
        return await apiClient.analyzeImage(file, options);
      case 'video':
        return await apiClient.analyzeVideo(file, options);
      case 'audio':
        return await apiClient.analyzeAudio(file, options);
      case 'multimodal':
        return await apiClient.analyzeMultimodal(file, options);
      default:
        throw new Error(`Unknown analysis type: ${analysisType}`);
    }
  }

  /**
   * Execute verification step
   */
  private async executeVerifyStep(step: TestStep): Promise<any> {
    const { condition, value, operator = 'equals' } = step.parameters;

    switch (operator) {
      case 'equals':
        if (condition !== value) {
          throw new Error(`Verification failed: expected ${value}, got ${condition}`);
        }
        break;
      case 'contains':
        if (!condition.includes(value)) {
          throw new Error(`Verification failed: ${condition} does not contain ${value}`);
        }
        break;
      case 'greaterThan':
        if (condition <= value) {
          throw new Error(`Verification failed: ${condition} is not greater than ${value}`);
        }
        break;
      case 'lessThan':
        if (condition >= value) {
          throw new Error(`Verification failed: ${condition} is not less than ${value}`);
        }
        break;
      default:
        throw new Error(`Unknown verification operator: ${operator}`);
    }

    return { verified: true, condition, value, operator };
  }

  /**
   * Execute wait step
   */
  private async executeWaitStep(step: TestStep): Promise<any> {
    const { duration, condition } = step.parameters;

    if (condition) {
      // Wait for condition to be met
      const startTime = Date.now();
      const timeout = step.timeout || 30000;

      while (Date.now() - startTime < timeout) {
        if (await this.checkCondition(condition)) {
          return { waited: Date.now() - startTime, condition: 'met' };
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      throw new Error(`Wait condition not met within ${timeout}ms`);
    } else {
      // Simple duration wait
      await new Promise(resolve => setTimeout(resolve, duration));
      return { waited: duration };
    }
  }

  /**
   * Execute cleanup step
   */
  private async executeCleanupStep(step: TestStep): Promise<any> {
    const { type, parameters } = step.parameters;

    switch (type) {
      case 'deleteFile':
        await fs.unlink(parameters.filePath).catch(() => {}); // Ignore errors
        return { deleted: parameters.filePath };
      case 'clearDatabase':
        // Implementation depends on your database setup
        return { cleared: 'database' };
      case 'resetState':
        // Reset application state
        return { reset: 'state' };
      default:
        throw new Error(`Unknown cleanup type: ${type}`);
    }
  }

  /**
   * Check a condition
   */
  private async checkCondition(condition: any): Promise<boolean> {
    // Implementation depends on the condition type
    // This is a simplified version
    if (typeof condition === 'function') {
      return await condition();
    }
    return Boolean(condition);
  }

  /**
   * Verify result against expected result
   */
  private verifyResult(actual: any, expected: any): void {
    if (typeof expected === 'object' && expected !== null) {
      for (const [key, value] of Object.entries(expected)) {
        if (actual[key] !== value) {
          throw new Error(`Expected ${key} to be ${value}, got ${actual[key]}`);
        }
      }
    } else if (actual !== expected) {
      throw new Error(`Expected ${expected}, got ${actual}`);
    }
  }

  /**
   * Run cleanup actions
   */
  private async runCleanup(cleanupActions: CleanupAction[]): Promise<void> {
    for (const action of cleanupActions) {
      try {
        await this.executeCleanupStep({
          name: `cleanup_${action.type}`,
          action: 'cleanup',
          parameters: action,
          timeout: 10000
        });
      } catch (error) {
        logger.warn('Cleanup action failed', {
          action: action.type,
          error: (error as Error).message
        });
      }
    }
  }

  /**
   * Get MIME type for file type
   */
  private getMimeType(fileType: string): string {
    const mimeTypes: Record<string, string> = {
      'image': 'image/jpeg',
      'video': 'video/mp4',
      'audio': 'audio/wav',
      'json': 'application/json'
    };
    return mimeTypes[fileType] || 'application/octet-stream';
  }

  /**
   * Get test execution status
   */
  getTestExecution(executionId: string): TestExecution | undefined {
    return this.runningTests.get(executionId);
  }

  /**
   * Get test result
   */
  getTestResult(executionId: string): TestResult | undefined {
    return this.testResults.get(executionId);
  }

  /**
   * Get all running tests
   */
  getRunningTests(): TestExecution[] {
    return Array.from(this.runningTests.values());
  }

  /**
   * Cancel a running test
   */
  cancelTest(executionId: string): boolean {
    const execution = this.runningTests.get(executionId);
    if (execution) {
      execution.status = 'cancelled';
      this.runningTests.delete(executionId);
      this.emit('testCancelled', { executionId });
      return true;
    }
    return false;
  }
}

interface TestExecution {
  id: string;
  scenario: TestScenario;
  startTime: Date;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  currentStep: number;
  stepResults: StepResult[];
}

// Export singleton instance
export const testOrchestrator = new TestOrchestrator();