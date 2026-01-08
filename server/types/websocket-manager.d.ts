import { Server as HttpServer } from 'http';

declare module '../services/websocket-manager' {
  export interface WebSocketManager {
    init: (server: HttpServer) => void;
    broadcast: (event: string, data: any) => void;
    // Add other methods as needed
  }

  export const webSocketManager: WebSocketManager;
}
