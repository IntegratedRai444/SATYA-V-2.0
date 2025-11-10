import { Request, Response, NextFunction } from 'express';
import { webSocketService } from '../services/websocket';

export const attachWebSocket = (req: Request, res: Response, next: NextFunction): void => {
  // Attach WebSocket service to the request object
  (req as any).ws = webSocketService;
  next();
};

export const emitToRoom = (req: Request, room: string, event: string, data: any): void => {
  const ws = (req as any).ws;
  if (ws) {
    ws.emitToRoom(room, event, data);
  }
};
