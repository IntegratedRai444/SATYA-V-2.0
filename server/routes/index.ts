import { Router, Request, Response } from 'express';
import { healthRouter } from './health.routes';
import analysisRouter from './analysis.routes';
import { dashboardRouter } from './dashboard.routes';
import { historyRouter } from './history';
import { chatRouter } from './chat';
import { notificationsRouter } from './notifications';
import { userRouter } from './user';
import { resultsRouter } from './results.routes';
import { modelsRouter } from './models.routes';
import { getSystemMetrics } from './system-metrics';
import { createApiError } from '../middleware/api-version';
import { requireAuth } from '../middleware/auth.middleware';
import { authRouter } from './auth';

const router = Router();

// Health check route (unversioned)
router.use('/health', healthRouter);

// API v2 Routes
const v2Router = Router();

// Auth routes (PUBLIC - for login, register, etc.)
v2Router.use('/auth', authRouter);

// Dashboard routes (PROTECTED)
v2Router.use('/dashboard', requireAuth, dashboardRouter);

// Test route without authentication
v2Router.get('/test', (req: Request, res: Response) => {
  res.json({ message: 'Test route working', timestamp: new Date().toISOString() });
});

// History routes (PROTECTED)
v2Router.use('/history', requireAuth, historyRouter);

// Analysis routes (PROTECTED)
v2Router.use('/analysis', requireAuth, analysisRouter);

// Chat routes (PROTECTED)
v2Router.use('/chat', requireAuth, chatRouter);

// Notifications routes (PROTECTED)
v2Router.use('/notifications', requireAuth, notificationsRouter);

// User routes (PROTECTED)
v2Router.use('/user', requireAuth, userRouter);

// Results routes (PROTECTED)
v2Router.use('/results', requireAuth, resultsRouter);

// Models routes (PUBLIC - for model info)
v2Router.use('/models', modelsRouter);

// System metrics route (PROTECTED - for admin monitoring)
v2Router.use('/system', requireAuth, getSystemMetrics);

// Note: WebSocket connections are handled directly by the WebSocket server
// at /api/v2/dashboard/ws - no Express route needed

// Versioned API routes
router.use('/', v2Router);

// Default version (v2)
router.use('/api', (req, res) => {
  // Redirect to the latest version
  const pathname = req.originalUrl.replace(/^\/api/, '');
  const newUrl = `/api/v2${pathname}`;
  
  // If this is a GET request, do a 302 redirect
  if (req.method === 'GET') {
    return res.redirect(302, newUrl);
  }
  
  // For non-GET requests, return a 400 with the correct URL
  res.status(400).json(createApiError(
    `API version is required. Use ${newUrl}`,
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

// REMOVED: Contract logger - No more deprecated routes
// REMOVED: Legacy redirect logic - Frontend must use canonical routes directly

// 404 handler for API routes (must be after all other routes)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
router.use('/api', (req: any, res: Response) => {
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
// eslint-disable-next-line @typescript-eslint/no-explicit-any
router.use((req: any, res: Response) => {
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

export { router as routes };
