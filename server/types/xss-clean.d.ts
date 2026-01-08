declare module 'xss-clean' {
  import { RequestHandler } from 'express';
  
  interface XssCleanOptions {
    // Add any options if needed
  }

  function xssClean(options?: XssCleanOptions): RequestHandler;
  
  export = xssClean;
}
