import { WebSocket } from 'ws';

export class TestWebSocketClient {
  private ws: WebSocket;
  private messageQueue: any[] = [];
  private messageHandlers: Map<string, Function[]> = new Map();

  constructor(url: string) {
    this.ws = new WebSocket(url);
    
    this.ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      this.messageQueue.push(message);
      
      // Call specific handlers if registered
      const handlers = this.messageHandlers.get(message.type) || [];
      handlers.forEach(handler => handler(message));
    });
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.on('open', () => resolve());
      this.ws.on('error', (err) => reject(err));
    });
  }

  close(): Promise<void> {
    return new Promise((resolve) => {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
        this.ws.on('close', () => resolve());
      } else {
        resolve();
      }
    });
  }

  send(message: any): void {
    this.ws.send(JSON.stringify(message));
  }

  waitForMessage(type: string, timeout = 5000): Promise<any> {
    return new Promise((resolve, reject) => {
      // Check if message is already in queue
      const existingMessage = this.messageQueue.find(m => m.type === type);
      if (existingMessage) {
        return resolve(existingMessage);
      }

      // Set up handler for new messages
      const handler = (message: any) => {
        if (message.type === type) {
          cleanup();
          resolve(message);
        }
      };

      // Set timeout
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(`Timeout waiting for message type: ${type}`));
      }, timeout);

      // Cleanup function
      const cleanup = () => {
        clearTimeout(timer);
        this.off(type, handler);
      };

      // Register handler
      this.on(type, handler);
    });
  }

  on(type: string, handler: (message: any) => void): void {
    const handlers = this.messageHandlers.get(type) || [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  off(type: string, handler: (message: any) => void): void {
    const handlers = this.messageHandlers.get(type) || [];
    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
      this.messageHandlers.set(type, handlers);
    }
  }
}

export async function createTestClient(url: string): Promise<TestWebSocketClient> {
  const client = new TestWebSocketClient(url);
  await client.connect();
  return client;
}
