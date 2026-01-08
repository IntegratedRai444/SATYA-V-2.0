import { Request, Response, NextFunction } from 'express';

const allowedOrigins = [
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  // Add your production domains here
  process.env.FRONTEND_URL,
  process.env.ADMIN_URL
].filter(Boolean) as string[];

const isAllowedOrigin = (origin: string): boolean => {
  if (process.env.NODE_ENV === 'development') {
    return true;
  }
  return allowedOrigins.some(allowedOrigin => 
    origin === allowedOrigin || 
    (process.env.NODE_ENV === 'production' && origin.endsWith(new URL(allowedOrigin).hostname))
  );
};

export const corsMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const origin = req.headers.origin;
  const requestMethod = req.method;

  // Handle CORS headers
  if (origin && isAllowedOrigin(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Vary', 'Origin');
  }

  // Handle preflight requests
  if (requestMethod === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, X-CSRF-Token');
    res.setHeader('Access-Control-Expose-Headers', 'X-Request-Id, X-Trace-Id, X-CSRF-Token');
    res.setHeader('Access-Control-Max-Age', '86400'); // 24 hours
    return res.status(204).end();
  }

  next();
};
