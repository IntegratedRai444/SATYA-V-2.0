/**
 * Response Standardization Utility
 * Ensures all API responses follow consistent format
 */

import { Request, Response, NextFunction } from 'express';

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

/**
 * Creates a successful API response
 */
export const createSuccessResponse = <T>(data: T, message?: string): ApiResponse<T> => ({
  success: true,
  data,
  message
});

/**
 * Creates an error API response
 */
export const createErrorResponse = (error: string, _statusCode: number = 500, message?: string): ApiResponse<never> => ({
  success: false,
  error,
  message
});

/**
 * Creates a paginated response
 */
export const createPaginatedResponse = <T>(
  data: T[], 
  meta: PaginationMeta, 
  message?: string
): ApiResponse<{ data: T[]; meta: PaginationMeta }> => ({
  success: true,
  data: { data, meta },
  message
});

/**
 * Standard response middleware
 */
export const standardizeResponse = (req: Request, res: Response, next: NextFunction) => {
  const originalJson = res.json;
  
  res.json = function(data: unknown) {
    // Ensure response follows standard format
    if (data && typeof data === 'object' && !Object.prototype.hasOwnProperty.call(data, 'success')) {
      // Response already standardized
      return originalJson.call(this, data);
    }
    
    // Auto-standardize responses
    return originalJson.call(this, data);
  };
  
  next();
};
