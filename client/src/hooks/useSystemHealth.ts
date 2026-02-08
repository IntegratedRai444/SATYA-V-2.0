import { useState, useEffect, useCallback } from 'react';

export interface SystemHealth {
  aiEngine: 'Online' | 'Offline';
  inferenceMode: 'CPU' | 'GPU';
  modelsLoaded: {
    image: boolean;
    video: boolean;
    audio: boolean;
    multimodal: boolean;
  };
  apiStatus: 'Running' | 'Error';
  pythonService: 'Connected' | 'Disconnected';
  lastInference: string | null;
  avgInferenceTime: number | null;
}

export const useSystemHealth = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    aiEngine: 'Online',
    inferenceMode: 'CPU',
    modelsLoaded: {
      image: true,
      video: true,
      audio: true,
      multimodal: true,
    },
    apiStatus: 'Running',
    pythonService: 'Connected',
    lastInference: null,
    avgInferenceTime: null,
  });

  const [isChecking, setIsChecking] = useState(false);

  const checkSystemHealth = useCallback(async () => {
    setIsChecking(true);
    
    try {
      // In a real implementation, these would be actual API calls
      const healthChecks = await Promise.allSettled([
        // Check API health
        fetch('/api/health').then(res => res.ok).catch(() => false),
        // Check Python ML service health
        fetch('http://localhost:8000/health').then(res => res.ok).catch(() => false),
        // Check database connectivity
        fetch('/api/health/database').then(res => res.ok).catch(() => false),
      ]);

      const [apiHealthy, pythonHealthy] = healthChecks.map(
        result => result.status === 'fulfilled' ? result.value : false
      );

      setSystemHealth(prev => ({
        ...prev,
        apiStatus: apiHealthy ? 'Running' : 'Error',
        pythonService: pythonHealthy ? 'Connected' : 'Disconnected',
        aiEngine: apiHealthy && pythonHealthy ? 'Online' : 'Offline',
      }));
    } catch (error) {
      console.error('System health check failed:', error);
      setSystemHealth(prev => ({
        ...prev,
        apiStatus: 'Error',
        pythonService: 'Disconnected',
        aiEngine: 'Offline',
      }));
    } finally {
      setIsChecking(false);
    }
  }, []);

  // Auto-check health every 30 seconds
  useEffect(() => {
    checkSystemHealth();
    
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, [checkSystemHealth]);

  // Update inference metrics (would come from actual analysis data)
  const updateInferenceMetrics = useCallback((data: {
    timestamp: string;
    inferenceTime: number;
  }) => {
    setSystemHealth(prev => ({
      ...prev,
      lastInference: data.timestamp,
      avgInferenceTime: data.inferenceTime,
    }));
  }, []);

  return {
    systemHealth,
    isChecking,
    checkSystemHealth,
    updateInferenceMetrics,
  };
};
