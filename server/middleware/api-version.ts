import { Request, Response, NextFunction } from 'express';

// Supported API versions
export const SUPPORTED_VERSIONS = ['v2'] as const;
export type ApiVersion = typeof SUPPORTED_VERSIONS[number];

export interface ApiError extends Error {
  status?: number;
  code?: string;
  details?: any;
}

// Default API version if none specified
const DEFAULT_API_VERSION: ApiVersion = 'v2';

/**
 * Middleware to handle API versioning
 */
export const versionMiddleware = (req: Request, res: Response, next: NextFunction) => {
  // Get version from Accept header or query parameter
  const acceptHeader = req.headers['accept'] || '';
  const versionParam = req.query['v'] as string;
  
  let version: string | null = null;
  
  // Check Accept header first (e.g., application/vnd.satyaai.v2+json)
  const versionMatch = acceptHeader.match(/application\/vnd\.satyaai\.(v\d+)\+json/);
  if (versionMatch) {
    version = versionMatch[1];
  }
  
  // Fall back to query parameter
  if (!version && versionParam) {
    version = versionParam.startsWith('v') ? versionParam : `v${versionParam}`;
  }
  
  // Default to latest version if none specified
  if (!version) {
    version = DEFAULT_API_VERSION;
  }
  
  // Validate version
  if (!SUPPORTED_VERSIONS.includes(version as ApiVersion)) {
    const error: ApiError = new Error(`Unsupported API version: ${version}`);
    error.status = 400;
    error.code = 'UNSUPPORTED_API_VERSION';
    return next(error);
  }
  
  // Set version on request for use in routes
  req.apiVersion = version as ApiVersion;
  
  // Set response headers
  res.set({
    'X-API-Version': version,
    'X-API-Version-Date': new Date().toISOString(),
    'X-API-Status': 'active'
  });
  
  next();
};

/**
 * 404 Handler for undefined routes
 */
export const notFoundHandler = (req: Request, res: Response) => {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `Cannot ${req.method} ${req.originalUrl}`,
      request_id: req.id,
      timestamp: new Date().toISOString()
    }
  });
};

/**
 * Global error handler
 */
export const errorHandler = (err: ApiError, req: Request, res: Response, next: NextFunction) => {
  // Log the error
  console.error(`[${new Date().toISOString()}] Error: ${err.message}\n${err.stack || ''}`);
  
  // Default error status
  const status = err.status || 500;
  const response: any = {
    error: {
      code: err.code || 'INTERNAL_SERVER_ERROR',
      message: status >= 500 ? 'Internal Server Error' : err.message,
      request_id: req.id,
      timestamp: new Date().toISOString()
    }
  };
  
  // Add error details in development
  if (process.env.NODE_ENV === 'development' && err.details) {
    response.error.details = err.details;
  }
  
  res.status(status).json(response);
};

/**
 * Create a custom API error
 */
export const createApiError = (
  message: string,
  status: number = 500,
  code: string = 'INTERNAL_SERVER_ERROR',
  details?: any
): ApiError => {
  const error = new Error(message) as ApiError;
  error.status = status;
  error.code = code;
  if (details) error.details = details;
  return error;
};
