import { Router } from 'express';

const router = Router();

// WebSocket upgrade handler for /api/v2/dashboard/ws
router.get('/api/v2/dashboard/ws', (req, res) => {
  res.status(426).json({
    error: 'WebSocket upgrade required',
    message: 'Please connect using WebSocket protocol',
    code: 'WEBSOCKET_REQUIRED'
  });
});

export { router as websocketRouter };
