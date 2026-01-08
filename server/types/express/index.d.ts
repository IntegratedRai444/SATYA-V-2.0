import { RequestHandler, ErrorRequestHandler } from 'express';

declare global {
  namespace Express {
    interface Request {
      id?: string;
      user?: {
        id: string;
        email: string;
        role: string;
      };
    }
  }
}

export {};
