import { Request, Response, NextFunction } from 'express';
import { validationResult, ValidationChain } from 'express-validator';
import { z, ZodError, ZodSchema, ZodTypeAny } from 'zod';
import { logger } from '../config/logger';

export interface ValidationErrorResponse {
  code: string;
  message: string;
  path?: (string | number)[];
  validation?: string;
}

export interface ValidationResult<T = any> {
  success: boolean;
  data?: T;
  errors?: ValidationErrorResponse[];
}

/**
 * Validates request data against a Zod schema
 */
export const validateSchema = <T extends ZodTypeAny>(schema: T) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      // Combine body, query, and params for validation
      const data = {
        body: req.body,
        query: req.query,
        params: req.params,
      };

      // Validate using Zod schema
      const result = schema.safeParse(data);

      if (!result.success) {
        const formattedErrors = result.error.issues.map(issue => ({
          code: issue.code,
          message: issue.message,
          path: issue.path,
          validation: issue.code,
        }));

        logger.warn('Schema validation failed', { errors: formattedErrors });

        return res.status(400).json({
          success: false,
          errors: formattedErrors,
        });
      }

      // Attach validated data to the request object
      req.validated = result.data;
      return next();
    } catch (error) {
      logger.error('Schema validation error:', error);
      return res.status(500).json({
        success: false,
        message: 'Internal server error during schema validation',
      });
    }
  };
};

/**
 * Middleware to validate request using express-validator
 */
export const validateRequest = (validations: ValidationChain[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      await Promise.all(validations.map(validation => validation.run(req)));

      const errors = validationResult(req);
      if (errors.isEmpty()) {
        return next();
      }

      const errorArray = errors.array();
      logger.warn('Request validation failed', { errors: errorArray });
      
      const formattedErrors = errorArray.map(err => {
        const param = 'param' in err ? (err as any).param : 'unknown';
        return {
          code: 'INVALID_INPUT',
          message: err.msg,
          path: [param],
          validation: err.msg,
        };
      });
      
      return res.status(400).json({
        success: false,
        errors: formattedErrors,
      });
    } catch (error) {
      logger.error('Request validation error:', error);
      return res.status(500).json({
        success: false,
        message: 'Internal server error during request validation',
      });
    }
  };
};

/**
 * Combined validation using both express-validator and Zod
 */
// Type for our validation middleware
type ValidationMiddleware = (req: Request, res: Response, next: NextFunction) => Promise<void> | void;

export const validate = <T extends ZodTypeAny>(
  validations: ValidationChain[],
  schema?: T
): Array<ValidationChain | ValidationMiddleware> => {
  // Create a new array with the validation chains
  const middlewareArray: Array<ValidationChain | ValidationMiddleware> = [...validations];
  
  // Add the validation middleware that runs the validations
  middlewareArray.push(validateRequest(validations) as unknown as ValidationMiddleware);
  
  // Add schema validation if provided
  if (schema) {
    middlewareArray.push(validateSchema(schema) as unknown as ValidationMiddleware);
  }
  
  return middlewareArray;
};

// Extend Express Request type to include validated data
declare global {
  namespace Express {
    interface Request {
      validated?: any;
    }
  }
}

export default validateRequest;
