import client from 'prom-client';

export const metricsRegistry = new (client as any).Registry();

// Enable default metrics (CPU, memory, etc.)
(client as any).collectDefaultMetrics({ register: metricsRegistry });

export const httpMetricsMiddleware = () => {
  return (req: any, res: any, next: () => void) => {
    const startTime = Date.now();
    
    res.on('finish', () => {
      const responseTime = Date.now() - startTime;
      // You can log or record metrics here
      console.log(`Response time: ${responseTime}ms`);
    });
    
    next();
  };
};
