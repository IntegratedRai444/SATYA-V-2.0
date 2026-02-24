import { Request, Response, NextFunction } from 'express';
import { validate as isUUID } from 'uuid';

/**
 * Middleware to validate job ID format
 * Prevents identity corruption by ensuring UUID format
 */
export function validateJobId(req: Request, res: Response, next: NextFunction) {
  const { jobId } = req.params;
  
  // Validate UUID format
  if (!isUUID(jobId)) {
    return res.status(400).json({
      success: false,
      error: 'INVALID_JOB_ID_FORMAT',
      message: `Job ID must be a valid UUID, received: ${jobId}`,
    });
  }
  
  next();
}
