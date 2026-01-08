import { Registry, collectDefaultMetrics } from 'prom-client';

export const metricsRegistry = new Registry();

// Enable default metrics (CPU, memory, etc.)
collectDefaultMetrics({ register: metricsRegistry });

export const httpMetricsMiddleware = () => {
  return (req: any, res: any, next: () => void) => {
    const startTime = Date.now();
    
    res.on('finish', () => {
      const responseTime = Date.now() - startTime;
      // You can log or record metrics here
    });
    
    next();
  };
};
