/**
 * Service Contract - Centralized Node ↔ Python Configuration
 * Prevents configuration drift and ensures consistent health checks
 */

export const SERVICE_CONTRACT = {
  // Python Service Configuration
  python: {
    // URL Configuration - Single Source of Truth
    baseUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:8000',
    
    // Health Check Contract
    health: {
      endpoint: '/health',
      expectedStatus: 'healthy',
      timeout: 5000, // 5 seconds
      interval: 30000, // 30 seconds
      requiredFields: ['status', 'timestamp'],
      successCodes: [200],
      failureCodes: [503, 504, 502]
    },
    
    // API Endpoints Contract
    endpoints: {
      analysis: '/api/v2/analysis/analyze/{media_type}',
      health: '/health',
      info: '/analyze/info'
    },
    
    // Timeout Configuration
    timeouts: {
      healthCheck: 5000,
      analysis: 120000, // 2 minutes
      connection: 10000
    },
    
    // Retry Configuration
    retry: {
      maxAttempts: 3,
      baseDelay: 1000,
      maxDelay: 30000,
      backoffFactor: 2
    }
  }
};

export default SERVICE_CONTRACT;
