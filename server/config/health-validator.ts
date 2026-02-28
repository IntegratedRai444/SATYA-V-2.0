/**
 * Enhanced Health Validator - Validates Python service health against contract
 * Prevents false positives and ensures proper service readiness
 */

import { SERVICE_CONTRACT } from './service-contract';
import { logger } from './logger';

// Node.js URL constructor
declare const URL: typeof globalThis.URL;

interface HealthResponse {
  status: string;
  timestamp: string;
  [key: string]: unknown;
}

interface ValidationResult {
  healthy: boolean;
  error?: string;
  responseTime: number;
  details?: Record<string, unknown>;
}

export class HealthValidator {
  /**
   * Validates Python service health against contract requirements
   */
  static validateHealthResponse(response: HealthResponse, responseTime: number): ValidationResult {
    const details: Record<string, unknown> = {
      responseTime,
      timestamp: response.timestamp,
      status: response.status
    };

    // Check required fields
    const missingFields = SERVICE_CONTRACT.python.health.requiredFields.filter(
      field => !(field in response)
    );
    
    if (missingFields.length > 0) {
      return {
        healthy: false,
        error: `Missing required health fields: ${missingFields.join(', ')}`,
        responseTime,
        details
      };
    }

    // Check status value
    if (response.status !== SERVICE_CONTRACT.python.health.expectedStatus) {
      return {
        healthy: false,
        error: `Invalid health status. Expected: '${SERVICE_CONTRACT.python.health.expectedStatus}', Got: '${response.status}'`,
        responseTime,
        details
      };
    }

    // Check response time
    if (responseTime > SERVICE_CONTRACT.python.health.timeout) {
      return {
        healthy: false,
        error: `Health check too slow: ${responseTime}ms > ${SERVICE_CONTRACT.python.health.timeout}ms`,
        responseTime,
        details
      };
    }

    logger.info('[HEALTH_VALIDATION] Service health validated', {
      status: response.status,
      responseTime,
      details
    });

    return {
      healthy: true,
      responseTime,
      details
    };
  }

  /**
   * Validates service configuration consistency
   */
  static validateConfiguration(): ValidationResult {
    const issues: string[] = [];

    // Check environment variables
    const pythonUrl = process.env.PYTHON_SERVICE_URL;
    if (!pythonUrl) {
      issues.push('PYTHON_SERVICE_URL not set');
    }

    // Check URL format
    if (pythonUrl && !pythonUrl.startsWith('http')) {
      issues.push('PYTHON_SERVICE_URL must start with http:// or https://');
    }

    // Check port consistency
    if (pythonUrl) {
      const url = new URL(pythonUrl);
      if (url.port !== '8000') {
        issues.push(`Python service port mismatch. Expected: 8000, Got: ${url.port}`);
      }
    }

    if (issues.length > 0) {
      return {
        healthy: false,
        error: `Configuration validation failed: ${issues.join(', ')}`,
        responseTime: 0,
        details: { issues }
      };
    }

    return {
      healthy: true,
      responseTime: 0,
      details: { configuration: 'valid' }
    };
  }
}

export default HealthValidator;
