import { api } from '@/lib/api/client';
import { useState, useEffect } from 'react';
import { metrics } from './metrics';

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: {
    api: boolean;
    database: boolean;
    mlService: boolean;
    storage: boolean;
  };
  metrics: {
    uptime: number;
    memoryUsage: {
      rss: number;
      heapTotal: number;
      heapUsed: number;
      external: number;
    };
    cpu: {
      user: number;
      system: number;
    };
  };
  details?: Record<string, any>;
}

class HealthService {
  private checkInterval: NodeJS.Timeout | null = null;
  private lastCheck: HealthCheckResult | null = null;
  private listeners: Array<(result: HealthCheckResult) => void> = [];
  private readonly CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.initialize();
  }

  private async initialize() {
    // Initial check
    await this.checkHealth();
    
    // Set up periodic checks
    this.checkInterval = setInterval(
      () => this.checkHealth(),
      this.CHECK_INTERVAL
    );
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', this.cleanup);
  }

  private cleanup = () => {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    window.removeEventListener('beforeunload', this.cleanup);
  };

  async checkHealth(): Promise<HealthCheckResult> {
    const startTime = performance.now();
    
    try {
      const result = await api.get<HealthCheckResult>('/api/health');
      this.lastCheck = result;
      
      // Track health check metrics
      const duration = performance.now() - startTime;
      metrics.track({
        type: 'health',
        name: 'health.check',
        value: duration,
        metadata: {
          status: result.status,
          services: result.services,
        },
      });

      // Notify listeners
      this.notifyListeners(result);
      return result;
    } catch (error) {
      const errorResult: HealthCheckResult = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        services: {
          api: false,
          database: false,
          mlService: false,
          storage: false,
        },
        metrics: {
          uptime: 0,
          memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0 },
          cpu: { user: 0, system: 0 },
        },
        details: {
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      };
      
      this.lastCheck = errorResult;
      this.notifyListeners(errorResult);
      
      // Track health check failure
      metrics.track({
        type: 'error',
        name: 'health.check.failed',
        value: 1,
        metadata: {
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      });
      
      return errorResult;
    }
  }

  getLastCheck(): HealthCheckResult | null {
    return this.lastCheck;
  }

  addListener(callback: (result: HealthCheckResult) => void): () => void {
    this.listeners.push(callback);
    
    // Return unsubscribe function
    return () => {
      this.listeners = this.listeners.filter(cb => cb !== callback);
    };
  }

  private notifyListeners(result: HealthCheckResult) {
    this.listeners.forEach(callback => {
      try {
        callback(result);
      } catch (error) {
        console.error('Error in health check listener:', error);
      }
    });
  }
}

// Singleton instance
export const healthService = new HealthService();

// React hook for components to use
export function useHealthCheck() {
  const [health, setHealth] = useState<HealthCheckResult | null>(() => 
    healthService.getLastCheck()
  );
  
  useEffect(() => {
    // Initial check
    healthService.checkHealth().then(setHealth);
    
    // Subscribe to updates
    const unsubscribe = healthService.addListener(setHealth);
    
    return () => {
      unsubscribe();
    };
  }, []);
  
  return {
    health,
    refresh: () => healthService.checkHealth().then(setHealth),
    isHealthy: health?.status === 'healthy',
    isLoading: health === null,
  };
}
