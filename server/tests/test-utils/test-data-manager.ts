import fs from 'fs/promises';
import path from 'path';
import { logger } from '../../config';

interface TestFile {
  name: string;
  type: 'image' | 'video' | 'audio';
  size: number;
  path: string;
  description: string;
  expectedResult?: {
    authenticity: 'real' | 'fake' | 'uncertain';
    confidence?: number;
  };
}

class TestDataManager {
  private testDataDir: string;
  private testFiles: Map<string, TestFile> = new Map();

  constructor() {
    this.testDataDir = path.join(__dirname, '../test-data');
    this.initializeTestData();
  }

  /**
   * Initialize test data
   */
  private async initializeTestData(): Promise<void> {
    try {
      await fs.access(this.testDataDir);
    } catch {
      await fs.mkdir(this.testDataDir, { recursive: true });
      await this.createSampleTestFiles();
    }

    await this.loadTestFileIndex();
    logger.info('Test data manager initialized', {
      testDataDir: this.testDataDir,
      testFiles: this.testFiles.size
    });
  }

  /**
   * Create sample test files
   */
  private async createSampleTestFiles(): Promise<void> {
    // Create sample image (1x1 pixel JPEG)
    const sampleImage = Buffer.from([
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
      0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
      0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
      0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
      0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
      0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01,
      0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
      0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFF, 0xC4,
      0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x0C,
      0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0x80, 0xFF, 0xD9
    ]);

    await fs.writeFile(path.join(this.testDataDir, 'sample-real-image.jpg'), sampleImage);

    // Create sample audio (minimal WAV header)
    const sampleAudio = Buffer.from([
      0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
      0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
      0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
      0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00
    ]);

    await fs.writeFile(path.join(this.testDataDir, 'sample-real-audio.wav'), sampleAudio);

    // Create test file index
    const testFileIndex = {
      'sample-real-image.jpg': {
        name: 'sample-real-image.jpg',
        type: 'image',
        size: sampleImage.length,
        path: 'sample-real-image.jpg',
        description: 'Sample real image for testing',
        expectedResult: {
          authenticity: 'real',
          confidence: 0.95
        }
      },
      'sample-real-audio.wav': {
        name: 'sample-real-audio.wav',
        type: 'audio',
        size: sampleAudio.length,
        path: 'sample-real-audio.wav',
        description: 'Sample real audio for testing',
        expectedResult: {
          authenticity: 'real',
          confidence: 0.90
        }
      }
    };

    await fs.writeFile(
      path.join(this.testDataDir, 'index.json'),
      JSON.stringify(testFileIndex, null, 2)
    );

    logger.info('Sample test files created', {
      files: Object.keys(testFileIndex)
    });
  }

  /**
   * Load test file index
   */
  private async loadTestFileIndex(): Promise<void> {
    try {
      const indexPath = path.join(this.testDataDir, 'index.json');
      const indexContent = await fs.readFile(indexPath, 'utf-8');
      const index = JSON.parse(indexContent);

      this.testFiles.clear();
      for (const [key, file] of Object.entries(index)) {
        this.testFiles.set(key, file as TestFile);
      }
    } catch (error) {
      logger.warn('Failed to load test file index', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Get test file
   */
  async getTestFile(fileName: string): Promise<Buffer> {
    const testFile = this.testFiles.get(fileName);
    if (!testFile) {
      throw new Error(`Test file not found: ${fileName}`);
    }

    const filePath = path.join(this.testDataDir, testFile.path);
    
    try {
      return await fs.readFile(filePath);
    } catch (error) {
      throw new Error(`Failed to read test file ${fileName}: ${(error as Error).message}`);
    }
  }

  /**
   * Get test file info
   */
  getTestFileInfo(fileName: string): TestFile | undefined {
    return this.testFiles.get(fileName);
  }

  /**
   * List available test files
   */
  listTestFiles(type?: 'image' | 'video' | 'audio'): TestFile[] {
    const files = Array.from(this.testFiles.values());
    return type ? files.filter(file => file.type === type) : files;
  }

  /**
   * Add custom test file
   */
  async addTestFile(
    fileName: string,
    fileBuffer: Buffer,
    metadata: Omit<TestFile, 'name' | 'size' | 'path'>
  ): Promise<void> {
    const filePath = path.join(this.testDataDir, fileName);
    await fs.writeFile(filePath, fileBuffer);

    const testFile: TestFile = {
      name: fileName,
      size: fileBuffer.length,
      path: fileName,
      ...metadata
    };

    this.testFiles.set(fileName, testFile);
    await this.saveTestFileIndex();

    logger.info('Test file added', {
      fileName,
      type: testFile.type,
      size: testFile.size
    });
  }

  /**
   * Remove test file
   */
  async removeTestFile(fileName: string): Promise<void> {
    const testFile = this.testFiles.get(fileName);
    if (!testFile) {
      return;
    }

    const filePath = path.join(this.testDataDir, testFile.path);
    
    try {
      await fs.unlink(filePath);
    } catch (error) {
      logger.warn('Failed to delete test file', {
        fileName,
        error: (error as Error).message
      });
    }

    this.testFiles.delete(fileName);
    await this.saveTestFileIndex();

    logger.info('Test file removed', { fileName });
  }

  /**
   * Save test file index
   */
  private async saveTestFileIndex(): Promise<void> {
    const index = Object.fromEntries(this.testFiles.entries());
    const indexPath = path.join(this.testDataDir, 'index.json');
    
    await fs.writeFile(indexPath, JSON.stringify(index, null, 2));
  }

  /**
   * Create test file from template
   */
  async createTestFileFromTemplate(
    template: 'real-image' | 'fake-image' | 'real-audio' | 'fake-audio' | 'real-video' | 'fake-video',
    fileName?: string
  ): Promise<string> {
    const timestamp = Date.now();
    const defaultFileName = fileName || `test-${template}-${timestamp}`;

    let fileBuffer: Buffer;
    let metadata: Omit<TestFile, 'name' | 'size' | 'path'>;

    switch (template) {
      case 'real-image':
        fileBuffer = await this.createSampleImage();
        metadata = {
          type: 'image',
          description: 'Generated real image for testing',
          expectedResult: { authenticity: 'real', confidence: 0.95 }
        };
        break;
      case 'fake-image':
        fileBuffer = await this.createSampleImage();
        metadata = {
          type: 'image',
          description: 'Generated fake image for testing',
          expectedResult: { authenticity: 'fake', confidence: 0.85 }
        };
        break;
      case 'real-audio':
        fileBuffer = await this.createSampleAudio();
        metadata = {
          type: 'audio',
          description: 'Generated real audio for testing',
          expectedResult: { authenticity: 'real', confidence: 0.90 }
        };
        break;
      case 'fake-audio':
        fileBuffer = await this.createSampleAudio();
        metadata = {
          type: 'audio',
          description: 'Generated fake audio for testing',
          expectedResult: { authenticity: 'fake', confidence: 0.80 }
        };
        break;
      default:
        throw new Error(`Unknown template: ${template}`);
    }

    await this.addTestFile(defaultFileName, fileBuffer, metadata);
    return defaultFileName;
  }

  /**
   * Create sample image
   */
  private async createSampleImage(): Promise<Buffer> {
    // Return the minimal JPEG we created earlier
    return Buffer.from([
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
      0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
      0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
      0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
      0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
      0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01,
      0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
      0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFF, 0xC4,
      0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x0C,
      0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0x80, 0xFF, 0xD9
    ]);
  }

  /**
   * Create sample audio
   */
  private async createSampleAudio(): Promise<Buffer> {
    // Return the minimal WAV we created earlier
    return Buffer.from([
      0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
      0x66, 0x6D, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
      0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
      0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00
    ]);
  }

  /**
   * Cleanup all test files
   */
  async cleanup(): Promise<void> {
    try {
      const files = await fs.readdir(this.testDataDir);
      for (const file of files) {
        if (file !== 'index.json') {
          await fs.unlink(path.join(this.testDataDir, file));
        }
      }
      this.testFiles.clear();
      await this.saveTestFileIndex();
      
      logger.info('Test data cleanup completed');
    } catch (error) {
      logger.error('Test data cleanup failed', {
        error: (error as Error).message
      });
    }
  }
}

// Export singleton instance
export const testDataManager = new TestDataManager();