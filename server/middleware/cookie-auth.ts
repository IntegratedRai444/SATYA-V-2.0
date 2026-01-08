import { Request, Response, NextFunction } from 'express';

/**
 * Middleware to handle cookie-based authentication
 */
export const cookieAuth = (req: Request, res: Response, next: NextFunction) => {
  // Check for session cookie
  if (req.cookies?.sessionId) {
    // Add user info to request if needed
    (req as any).user = { id: req.cookies.userId };
    return next();
  }
  
  // If no session cookie, return 401
  res.status(401).json({ error: 'Unauthorized - No valid session' });
};

export default cookieAuth;
