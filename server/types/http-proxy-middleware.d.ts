declare module 'http-proxy-middleware' {
  import { RequestHandler } from 'express';
  
  interface Options {
    target?: string;
    changeOrigin?: boolean;
    pathRewrite?: { [key: string]: string } | Function;
    onProxyReq?: (proxyReq: any, req: any, res: any) => void;
    onError?: (err: Error, req: any, res: any) => void;
  }

  export function createProxyMiddleware(options: Options): RequestHandler;
  
  // For older versions that use the factory pattern
  export default function createProxyMiddleware(
    options: Options
  ): RequestHandler;
}

declare module 'http-proxy-middleware/dist/types' {
  export * from 'http-proxy-middleware';
}
