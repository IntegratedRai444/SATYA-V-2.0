import { Router, Request, Response, NextFunction } from 'express';
import { healthRouter } from './health.routes';
import { authRouter } from './auth.routes';
import { analysisRouter } from './analysis.routes';
import historyRouter from './history';
import chatRouter from './chat';
import { createApiError } from '../middleware/api-version';

const router = Router();

// Health check route (unversioned)
router.use('/health', healthRouter);

// API v2 Routes
const v2Router = Router();

// Authentication routes
v2Router.use('/auth', authRouter);

// History routes
v2Router.use('/history', historyRouter);

// Analysis routes
v2Router.use('/analyze', analysisRouter);

// Chat routes
v2Router.use('/chat', chatRouter);

// Versioned API routes
router.use('/api/v2', v2Router);

// Default version (v2)
router.use('/api', (req, res, next) => {
  // Redirect to the latest version
  const url = new URL(req.originalUrl, 'http://dummy.com');
  url.pathname = `/api/v2${url.pathname.replace(/^\/api/, '')}`;
  
  // If this is a GET request, do a 302 redirect
  if (req.method === 'GET') {
    return res.redirect(302, url.pathname + url.search);
  }
  
  // For non-GET requests, return a 400 with the correct URL
  next(createApiError(
    `API version is required. Use /api/v2${req.path}`,
    400,
    'API_VERSION_REQUIRED',
    { 
      current_version: 'v2',
      supported_versions: ['v2'],
      docs: 'https://docs.satyaai.com/api/versions'
    }
  ));
});

// Root path redirect
router.get('/', (req: Request, res: Response) => {
  res.redirect('/api');
});

// 404 handler for API routes (must be after all other routes)
router.use('/api', (req: Request, res: Response) => {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `Cannot ${req.method} ${req.originalUrl}`,
      request_id: req.id,
      timestamp: new Date().toISOString(),
      docs: 'https://docs.satyaai.com/api/errors/NOT_FOUND'
    }
  });
});

// 404 handler for all other routes (must be the last route)
router.use((req: Request, res: Response) => {
  res.status(404).send(`
    <!DOCTYPE html>
    <html>
      <head>
        <title>404 Not Found</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
          h1 { color: #333; }
          p { color: #666; }
          a { color: #0066cc; text-decoration: none; }
        </style>
      </head>
      <body>
        <h1>404 - Page Not Found</h1>
        <p>The requested URL ${req.originalUrl} was not found on this server.</p>
        <p><a href="/">Go to Homepage</a> | <a href="/api">API Documentation</a></p>
      </body>
    </html>
  `);
});

export { router };
