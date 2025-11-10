import { PythonShell } from 'python-shell';
import path from 'path';
import { logger } from '../config';
import { promisify } from 'util';
import { EventEmitter } from 'events';

type PythonResponse = {
  success: boolean;
  result?: any;
  error?: string;
};

type PythonBridgeOptions = {
  pythonPath?: string;
  scriptPath?: string;
  modelsDir?: string;
  autoInitialize?: boolean;
  timeout?: number;
  maxBufferSize?: number;
};

export class PythonBridge {
  private static instance: PythonBridge;
  private pythonShell: PythonShell | null = null;
  private isInitialized = false;
  private options: Required<PythonBridgeOptions>;
  private eventEmitter = new EventEmitter();
  private retryCount = 0;
  private readonly MAX_RETRIES = 3;
  private readonly RETRY_DELAY = 2000; // 2 seconds
  private messageQueue: Array<{command: string; args: any[]; resolve: (value: any) => void; reject: (reason?: any) => void}> = [];
  private isProcessingQueue = false;

  private defaultOptions: Required<PythonBridgeOptions> = {
    pythonPath: process.env.PYTHON_PATH || 'python',
    scriptPath: path.join(__dirname, '../../python'),
    modelsDir: path.join(__dirname, '../../models'),
    autoInitialize: true,
    timeout: 300000, // 5 minutes
    maxBufferSize: 50 * 1024 * 1024 // 50MB
  };

  private pythonPath: string;
  private scriptPath: string;
  private modelsDir: string;
  private messageHandlers = new Map<string, (data: any) => void>();
  private pendingRequests = new Map<string, { 
    resolve: (value: any) => void; 
    reject: (reason?: any) => void; 
  }>();

  private constructor(options: PythonBridgeOptions = {}) {
    this.options = { ...this.defaultOptions, ...options };
    
    // Initialize required properties
    this.pythonPath = this.options.pythonPath;
    this.scriptPath = this.options.scriptPath;
    this.modelsDir = this.options.modelsDir;
    
    if (this.options.autoInitialize) {
      this.initialize().catch(error => {
        logger.error('Failed to initialize Python bridge:', error);
      });
    }
  }

  public static getInstance(options: PythonBridgeOptions = {}): PythonBridge {
    if (!PythonBridge.instance) {
      PythonBridge.instance = new PythonBridge(options);
    }
    return PythonBridge.instance;
  }

  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      logger.warn('Python bridge already initialized');
      return;
    }

    try {
      this.pythonShell = new PythonShell(this.scriptPath, {
        mode: 'json',
        pythonPath: this.pythonPath,
        pythonOptions: ['-u'],
        args: ['--models-dir', this.modelsDir]
      });

      // Set up message handler
      this.pythonShell.on('message', this.handlePythonMessage.bind(this));
      this.pythonShell.on('error', this.handlePythonError.bind(this));
      this.pythonShell.on('close', this.handlePythonClose.bind(this));

      // Wait for initialization confirmation
      await this.waitForInitialization();
      this.isInitialized = true;
      logger.info('Python bridge initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Python bridge:', error);
      await this.cleanup();
      throw error;
    }
  }

  private async waitForInitialization(timeout = 10000): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('Python bridge initialization timeout'));
      }, timeout);

      const checkInitialized = (message: any) => {
        if (message?.type === 'initialized') {
          clearTimeout(timer);
          this.pythonShell?.removeListener('message', checkInitialized);
          resolve();
        }
      };

      this.pythonShell?.on('message', checkInitialized);
    });
  }

  private handlePythonMessage(message: any): void {
    try {
      // Handle JSON messages
      if (typeof message === 'string') {
        message = JSON.parse(message);
      }

      // Handle request/response pattern
      if (message.id && this.pendingRequests.has(message.id)) {
        const { resolve, reject } = this.pendingRequests.get(message.id)!;
        this.pendingRequests.delete(message.id);
        
        if (message.error) {
          reject(new Error(message.error));
        } else {
          resolve(message.result);
        }
        return;
      }

      // Handle event-based messages
      if (message.event) {
        const handler = this.messageHandlers.get(message.event);
        if (handler) {
          handler(message.data);
        } else if (message.event !== 'heartbeat') {  // Ignore heartbeat if no handler
          logger.warn(`No handler registered for event: ${message.event}`);
        }
        return;
      }

      logger.warn('Received unhandled message:', message);
    } catch (error) {
      logger.error('Error handling Python message:', error);
    }
  }

  private handlePythonError(error: Error): void {
    logger.error('Python process error:', error);
    this.cleanup().catch(cleanupError => {
      logger.error('Error during cleanup after Python process error:', cleanupError);
    });
  }

  private handlePythonClose(code: number, signal: string | null): void {
    logger.warn(`Python process exited with code ${code} and signal ${signal}`);
    this.cleanup().catch(error => {
      logger.error('Error during cleanup after Python process close:', error);
    });
  }

  public async execute<T = any>(
    method: string,
    params: Record<string, any> = {},
    timeout = 30000
  ): Promise<T> {
    if (!this.pythonShell || !this.isInitialized) {
      throw new Error('Python bridge not initialized');
    }

    return new Promise((resolve, reject) => {
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
      const message = { id: requestId, method, params };
      
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Python bridge request timed out after ${timeout}ms`));
      }, timeout);

      this.pendingRequests.set(requestId, {
        resolve: (result) => {
          clearTimeout(timeoutId);
          resolve(result);
        },
        reject: (error) => {
          clearTimeout(timeoutId);
          reject(error);
        }
      });

      try {
        this.pythonShell?.send(JSON.stringify(message));
      } catch (error) {
        this.pendingRequests.delete(requestId);
        clearTimeout(timeoutId);
        reject(error);
      }
    });
  }

  public on(event: string, handler: (data: any) => void): () => void {
    this.messageHandlers.set(event, handler);
    return () => this.messageHandlers.delete(event);
  }

  public off(event: string): void {
    this.messageHandlers.delete(event);
  }

  public async startPythonServer(port = 5000): Promise<void> {
    return this.execute('start_server', { port });
  }

  public async stopPythonServer(): Promise<void> {
    return this.execute('stop_server');
  }

  public async restart(): Promise<void> {
    logger.info('Restarting Python bridge...');
    await this.shutdown();
    await this.initialize();
  }

  public async shutdown(): Promise<void> {
    if (!this.pythonShell) return;

    try {
      await this.execute('shutdown');
    } finally {
      await this.cleanup();
    }
  }

  private async cleanup(): Promise<void> {
    if (this.pythonShell) {
      try {
        // Reject all pending requests
        for (const [id, { reject }] of this.pendingRequests.entries()) {
          reject(new Error('Python bridge shutdown'));
          this.pendingRequests.delete(id);
        }

        // End the Python shell
        this.pythonShell.end(() => {
          logger.debug('Python shell ended');
        });
      } catch (error) {
        logger.error('Error during Python bridge cleanup:', error);
      } finally {
        this.pythonShell = null;
        this.isInitialized = false;
      }
    }
  }

  // Add any additional methods needed for your Python bridge
  public async getModelInfo(modelId: string): Promise<any> {
    return this.execute('get_model_info', { model_id: modelId });
  }

  public async predict(input: any, modelId?: string): Promise<any> {
    return this.execute('predict', { input, model_id: modelId });
  }
}

}

// Create and export a singleton instance
const pythonBridge = PythonBridge.getInstance();

// For backward compatibility
export { pythonBridge as pythonBridgeEnhanced };

export { pythonBridge };
export default pythonBridge;

// Initialize on import if not in test environment
if (process.env.NODE_ENV !== 'test') {
  pythonBridge.initialize().catch(error => {
    logger.error('Failed to initialize Python bridge on import:', error);
  });
}
