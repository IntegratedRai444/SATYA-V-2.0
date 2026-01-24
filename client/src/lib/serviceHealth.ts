/**
 * Service Health Checker
 * Detects if backend services are running without making API calls
 */

export interface ServiceStatus {
  nodeBackend: 'running' | 'stopped' | 'unknown';
  pythonML: 'running' | 'stopped' | 'unknown';
  frontend: 'running' | 'stopped' | 'unknown';
}

export interface ServiceHealthCheck {
  isHealthy: boolean;
  services: ServiceStatus;
  issues: string[];
  recommendations: string[];
}

/**
 * Check if services are likely running based on configuration
 * This is a static check - doesn't actually ping the services
 */
export function checkServiceHealth(): ServiceHealthCheck {
  const issues: string[] = [];
  const recommendations: string[] = [];
  const services: ServiceStatus = {
    nodeBackend: 'unknown',
    pythonML: 'unknown', 
    frontend: 'unknown'
  };

  // Check if we're in development mode
  const isDevelopment = import.meta.env.DEV;
  
  // Check environment variables
  const nodeUrl = import.meta.env.VITE_API_URL || 'http://localhost:5001';
  const pythonUrl = import.meta.env.VITE_ANALYSIS_API_URL || 'http://localhost:5001';
  
  // Frontend is running if this code executes
  services.frontend = 'running';
  
  // Node backend status (based on environment config)
  if (nodeUrl.includes('localhost:5001')) {
    services.nodeBackend = isDevelopment ? 'stopped' : 'unknown';
    if (isDevelopment) {
      issues.push('Node backend not detected on port 5001');
      recommendations.push('Start Node backend: cd server && npm run dev');
    }
  } else {
    services.nodeBackend = 'unknown';
  }
  
  // Python ML status (inferred from config)
  if (pythonUrl.includes('localhost:5001')) {
    // If Python is proxied through Node, we can't detect it separately
    services.pythonML = 'unknown';
  } else if (pythonUrl.includes('localhost:8000')) {
    services.pythonML = isDevelopment ? 'stopped' : 'unknown';
    if (isDevelopment) {
      issues.push('Python ML service not detected on port 8000');
      recommendations.push('Start Python ML: cd server/python && uvicorn main_api:app --reload');
    }
  }

  const isHealthy = issues.length === 0;

  return {
    isHealthy,
    services,
    issues,
    recommendations
  };
}

/**
 * Get user-friendly error message based on service health
 */
export function getServiceErrorMessage(error: Error): string {
  const health = checkServiceHealth();
  
  if (error.message.includes('Network Error') || error.message.includes('ERR_CONNECTION_REFUSED')) {
    if (health.services.nodeBackend === 'stopped') {
      return 'Backend services are not running. Please start the Node.js backend on port 5001.';
    }
    return 'Cannot connect to backend services. Please check if all services are running.';
  }
  
  if (error.message.includes('timeout')) {
    return 'Request timed out. The services may be overloaded or not responding.';
  }
  
  return error.message;
}

/**
 * Log service health status for debugging
 */
export function logServiceHealth(): void {
  const health = checkServiceHealth();
  
  console.group('🔍 Service Health Check');
  console.log('Overall Status:', health.isHealthy ? '✅ Healthy' : '❌ Issues Detected');
  console.log('Frontend:', health.services.frontend);
  console.log('Node Backend:', health.services.nodeBackend);
  console.log('Python ML:', health.services.pythonML);
  
  if (health.issues.length > 0) {
    console.warn('Issues:', health.issues);
  }
  
  if (health.recommendations.length > 0) {
    console.info('Recommendations:', health.recommendations);
  }
  
  console.groupEnd();
}
