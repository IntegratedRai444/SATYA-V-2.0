import { Request, Response, NextFunction } from 'express';
import { logger } from './logger';

const logAudit = (data: Record<string, any>) => {
  logger.info('AUDIT', data);
};

export const auditLogger = {
  log: logAudit,
  
  auditMiddleware: () => {
    return (req: Request, res: Response, next: NextFunction) => {
      const startTime = Date.now();
      
      res.on('finish', () => {
        const auditData = {
          timestamp: new Date().toISOString(),
          method: req.method,
          url: req.originalUrl,
          statusCode: res.statusCode,
          responseTime: Date.now() - startTime,
          ip: req.ip,
          userAgent: req.get('user-agent'),
          userId: (req as any).user?.id || 'anonymous',
        };
        
        logAudit(auditData);
      });
      
      next();
    };
  }
};

export default auditLogger;
