import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../config/logger';
import { circuitBreaker } from '../utils/circuitBreaker';

export interface FileMetadata {
  id: string;
  originalName: string;
  mimeType: string;
  size: number;
  uploadDate: Date;
  userId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  metadata?: Record<string, any>;
}

export interface FileProcessorOptions {
  uploadDir: string;
  maxFileSize: number;
  allowedMimeTypes: string[];
  maxConcurrentProcesses: number;
  cleanupAfterDays: number;
}

const DEFAULT_OPTIONS: FileProcessorOptions = {
  uploadDir: './uploads',
  maxFileSize: 1024 * 1024 * 100, // 100MB
  allowedMimeTypes: [
    'image/jpeg',
    'image/png',
    'application/pdf',
    'text/plain',
    'application/json'
  ],
  maxConcurrentProcesses: 5,
  cleanupAfterDays: 7
};

export interface ProcessingStats {
  activeProcesses: number;
  queuedFiles: number;
  processedFiles: number;
}

export class FileProcessor extends EventEmitter {
  private _activeProcesses = 0;
  private processingQueue: Array<() => Promise<void>> = [];
  private processedFilesCount = 0;
  private options: FileProcessorOptions;

  get activeProcesses(): number {
    return this._activeProcesses;
  }

  getStats(): ProcessingStats {
    return {
      activeProcesses: this._activeProcesses,
      queuedFiles: this.processingQueue.length,
      processedFiles: this.processedFilesCount
    };
  }
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(options: Partial<FileProcessorOptions> = {}) {
    super();
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      // Ensure upload directory exists
      await fs.mkdir(this.options.uploadDir, { recursive: true });
      
      // Start cleanup scheduler
      this.scheduleCleanup();
      
      logger.info('File processor initialized', {
        uploadDir: this.options.uploadDir,
        maxFileSize: this.options.maxFileSize,
        allowedMimeTypes: this.options.allowedMimeTypes
      });
    } catch (error) {
      logger.error('Failed to initialize file processor', { error });
      throw error;
    }
  }

  public async processFile(
    file: Buffer,
    metadata: Omit<FileMetadata, 'id' | 'uploadDate' | 'status'>
  ): Promise<FileMetadata> {
    const fileMetadata: FileMetadata = {
      ...metadata,
      id: uuidv4(),
      uploadDate: new Date(),
      status: 'processing'
    };

    try {
      // Validate file
      this.validateFile(file, fileMetadata);
      
      // Process the file (e.g., save to disk, process in background)
      await this.saveFile(file, fileMetadata);
      
      // Update status
      fileMetadata.status = 'completed';
      this.emit('processed', fileMetadata);
      
      return fileMetadata;
    } catch (error) {
      fileMetadata.status = 'failed';
      fileMetadata.error = error instanceof Error ? error.message : 'Unknown error';
      this.emit('error', error, fileMetadata);
      throw error;
    }
  }

  private validateFile(file: Buffer, metadata: FileMetadata): void {
    // Check file size
    if (file.length > this.options.maxFileSize) {
      throw new Error(`File size exceeds maximum allowed size of ${this.options.maxFileSize} bytes`);
    }

    // Check MIME type
    if (!this.options.allowedMimeTypes.includes(metadata.mimeType)) {
      throw new Error(`Unsupported file type: ${metadata.mimeType}`);
    }
  }

  private async saveFile(file: Buffer, metadata: FileMetadata): Promise<void> {
    const filePath = this.getFilePath(metadata);
    
    try {
      // Use circuit breaker for file operations
      await circuitBreaker.execute(
        () => fs.writeFile(filePath, file),
        { operationTimeout: 30000 } // 30s timeout for file operations
      );
      
      logger.info('File saved successfully', {
        fileId: metadata.id,
        filePath,
        size: file.length
      });
    } catch (error) {
      logger.error('Failed to save file', {
        fileId: metadata.id,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  private getFilePath(metadata: FileMetadata): string {
    const fileExt = path.extname(metadata.originalName) || this.getExtensionFromMime(metadata.mimeType);
    const fileName = `${metadata.id}${fileExt}`;
    return path.join(this.options.uploadDir, fileName);
  }

  private getExtensionFromMime(mimeType: string): string {
    const extMap: Record<string, string> = {
      'image/jpeg': '.jpg',
      'image/png': '.png',
      'application/pdf': '.pdf',
      'text/plain': '.txt',
      'application/json': '.json'
    };
    return extMap[mimeType] || '.bin';
  }

  private scheduleCleanup(): void {
    // Run cleanup every hour
    this.cleanupInterval = setInterval(
      () => this.cleanupOldFiles().catch(error => 
        logger.error('Failed to clean up old files', { error })
      ),
      60 * 60 * 1000 // 1 hour
    );
  }

  private async cleanupOldFiles(): Promise<void> {
    try {
      const now = new Date();
      const cutoffDate = new Date();
      cutoffDate.setDate(now.getDate() - this.options.cleanupAfterDays);

      const files = await fs.readdir(this.options.uploadDir);
      
      for (const file of files) {
        const filePath = path.join(this.options.uploadDir, file);
        const stats = await fs.stat(filePath);
        
        if (stats.mtime < cutoffDate) {
          await fs.unlink(filePath);
          logger.debug('Cleaned up old file', { filePath, lastModified: stats.mtime });
        }
      }
    } catch (error) {
      logger.error('Error during file cleanup', { error });
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    
    // Wait for active processes to complete with a timeout
    await Promise.race([
      this.waitForProcesses(),
      new Promise(resolve => setTimeout(resolve, 10000)) // 10s timeout
    ]);
  }

  private async waitForProcesses(): Promise<void> {
    while (this.activeProcesses > 0) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
}

// Export singleton instance
export const fileProcessor = new FileProcessor();

// Handle process termination
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down file processor...');
  await fileProcessor.shutdown();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down file processor...');
  await fileProcessor.shutdown();
  process.exit(0);
});

export default FileProcessor;
