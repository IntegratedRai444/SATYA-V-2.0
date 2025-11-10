import fs from 'fs/promises';
import path from 'path';
import { logger, config } from '../config';

interface CleanupOptions {
  maxAge?: number; // Maximum age in milliseconds
  directory?: string;
  pattern?: RegExp;
  dryRun?: boolean;
}

class FileCleanupService {
  private cleanupInterval: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor() {
    // Start automatic cleanup if configured
    if (config.CLEANUP_INTERVAL > 0) {
      this.startAutomaticCleanup();
    }
  }

  /**
   * Start automatic cleanup process
   */
  startAutomaticCleanup(): void {
    if (this.cleanupInterval) {
      return; // Already running
    }

    logger.info('Starting automatic file cleanup service', {
      interval: config.CLEANUP_INTERVAL,
      uploadDir: config.UPLOAD_DIR
    });

    this.cleanupInterval = setInterval(async () => {
      try {
        await this.cleanupOldFiles({
          directory: config.UPLOAD_DIR,
          maxAge: 24 * 60 * 60 * 1000 // 24 hours
        });
      } catch (error) {
        logger.error('Automatic cleanup failed', {
          error: (error as Error).message
        });
      }
    }, config.CLEANUP_INTERVAL);
  }

  /**
   * Stop automatic cleanup process
   */
  stopAutomaticCleanup(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
      logger.info('Stopped automatic file cleanup service');
    }
  }

  /**
   * Clean up old files in directory
   */
  async cleanupOldFiles(options: CleanupOptions = {}): Promise<{
    deletedCount: number;
    totalSize: number;
    errors: string[];
  }> {
    const {
      maxAge = 24 * 60 * 60 * 1000, // 24 hours default
      directory = config.UPLOAD_DIR,
      pattern = /.*/, // Match all files by default
      dryRun = false
    } = options;

    if (this.isRunning) {
      logger.warn('Cleanup already in progress, skipping');
      return { deletedCount: 0, totalSize: 0, errors: [] };
    }

    this.isRunning = true;
    const results = {
      deletedCount: 0,
      totalSize: 0,
      errors: [] as string[]
    };

    try {
      logger.info('Starting file cleanup', {
        directory,
        maxAge: `${maxAge / 1000 / 60}min`,
        dryRun
      });

      // Check if directory exists
      try {
        await fs.access(directory);
      } catch (error) {
        logger.warn('Upload directory does not exist', { directory });
        return results;
      }

      const files = await fs.readdir(directory);
      const now = Date.now();

      for (const filename of files) {
        try {
          // Skip if filename doesn't match pattern
          if (!pattern.test(filename)) {
            continue;
          }

          const filePath = path.join(directory, filename);
          const stats = await fs.stat(filePath);

          // Skip directories
          if (!stats.isFile()) {
            continue;
          }

          // Check if file is old enough
          const fileAge = now - stats.mtime.getTime();
          if (fileAge > maxAge) {
            results.totalSize += stats.size;

            if (!dryRun) {
              await fs.unlink(filePath);
              logger.debug('Deleted old file', {
                filename,
                age: `${Math.round(fileAge / 1000 / 60)}min`,
                size: stats.size
              });
            } else {
              logger.debug('Would delete old file (dry run)', {
                filename,
                age: `${Math.round(fileAge / 1000 / 60)}min`,
                size: stats.size
              });
            }

            results.deletedCount++;
          }
        } catch (error) {
          const errorMsg = `Failed to process file ${filename}: ${(error as Error).message}`;
          results.errors.push(errorMsg);
          logger.error('File cleanup error', {
            filename,
            error: (error as Error).message
          });
        }
      }

      logger.info('File cleanup completed', {
        directory,
        deletedCount: results.deletedCount,
        totalSize: `${Math.round(results.totalSize / 1024 / 1024 * 100) / 100}MB`,
        errors: results.errors.length,
        dryRun
      });

    } catch (error) {
      logger.error('File cleanup failed', {
        error: (error as Error).message,
        directory
      });
      results.errors.push(`Cleanup failed: ${(error as Error).message}`);
    } finally {
      this.isRunning = false;
    }

    return results;
  }

  /**
   * Clean up specific file
   */
  async cleanupFile(filePath: string): Promise<boolean> {
    try {
      await fs.unlink(filePath);
      logger.debug('File cleaned up successfully', { filePath });
      return true;
    } catch (error) {
      if ((error as any).code === 'ENOENT') {
        // File doesn't exist, consider it cleaned up
        return true;
      }
      
      logger.error('Failed to cleanup file', {
        error: (error as Error).message,
        filePath
      });
      return false;
    }
  }

  /**
   * Clean up files by pattern
   */
  async cleanupByPattern(directory: string, pattern: RegExp, maxAge?: number): Promise<{
    deletedCount: number;
    totalSize: number;
    errors: string[];
  }> {
    return this.cleanupOldFiles({
      directory,
      pattern,
      maxAge
    });
  }

  /**
   * Get directory statistics
   */
  async getDirectoryStats(directory: string = config.UPLOAD_DIR): Promise<{
    fileCount: number;
    totalSize: number;
    oldestFile: Date | null;
    newestFile: Date | null;
    averageAge: number;
  }> {
    const stats = {
      fileCount: 0,
      totalSize: 0,
      oldestFile: null as Date | null,
      newestFile: null as Date | null,
      averageAge: 0
    };

    try {
      const files = await fs.readdir(directory);
      const now = Date.now();
      let totalAge = 0;

      for (const filename of files) {
        try {
          const filePath = path.join(directory, filename);
          const fileStats = await fs.stat(filePath);

          if (fileStats.isFile()) {
            stats.fileCount++;
            stats.totalSize += fileStats.size;
            
            const fileAge = now - fileStats.mtime.getTime();
            totalAge += fileAge;

            if (!stats.oldestFile || fileStats.mtime < stats.oldestFile) {
              stats.oldestFile = fileStats.mtime;
            }

            if (!stats.newestFile || fileStats.mtime > stats.newestFile) {
              stats.newestFile = fileStats.mtime;
            }
          }
        } catch (error) {
          logger.debug('Error reading file stats', {
            filename,
            error: (error as Error).message
          });
        }
      }

      if (stats.fileCount > 0) {
        stats.averageAge = totalAge / stats.fileCount;
      }

    } catch (error) {
      logger.error('Failed to get directory stats', {
        error: (error as Error).message,
        directory
      });
    }

    return stats;
  }

  /**
   * Emergency cleanup - remove all files older than specified age
   */
  async emergencyCleanup(maxAge: number = 60 * 60 * 1000): Promise<{
    deletedCount: number;
    totalSize: number;
    errors: string[];
  }> {
    logger.warn('Starting emergency cleanup', {
      maxAge: `${maxAge / 1000 / 60}min`,
      directory: config.UPLOAD_DIR
    });

    return this.cleanupOldFiles({
      directory: config.UPLOAD_DIR,
      maxAge,
      pattern: /.*/ // Clean all files
    });
  }

  /**
   * Shutdown cleanup service
   */
  async shutdown(): Promise<void> {
    this.stopAutomaticCleanup();
    
    // Wait for any running cleanup to finish
    while (this.isRunning) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    logger.info('File cleanup service shutdown completed');
  }
}

// Export singleton instance
export const fileCleanupService = new FileCleanupService();

// Graceful shutdown handlers
process.on('SIGTERM', async () => {
  await fileCleanupService.shutdown();
});

process.on('SIGINT', async () => {
  await fileCleanupService.shutdown();
});