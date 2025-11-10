import { Server } from 'http';
import { AddressInfo } from 'net';
import { createServer } from 'http';
import { app } from '../../app'; // Assuming your Express app is exported from app.ts

let testServer: Server;
let testServerUrl: string;

export async function startTestServer(port = 0): Promise<string> {
  return new Promise((resolve, reject) => {
    testServer = createServer(app).listen(port, '0.0.0.0', () => {
      const address = testServer.address() as AddressInfo;
      testServerUrl = `http://localhost:${address.port}`;
      console.log(`Test server running at ${testServerUrl}`);
      resolve(testServerUrl);
    });
    
    testServer.on('error', reject);
  });
}

export async function stopTestServer(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!testServer) {
      return resolve();
    }
    
    testServer.close((err) => {
      if (err) {
        console.error('Error closing test server:', err);
        return reject(err);
      }
      console.log('Test server stopped');
      resolve();
    });
  });
}

export function getTestServerUrl(): string {
  if (!testServerUrl) {
    throw new Error('Test server not started');
  }
  return testServerUrl;
}
