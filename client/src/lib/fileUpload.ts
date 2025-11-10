import apiClient from './api';

export interface UploadProgress {
  fileId: string;
  fileName: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
  result?: any;
}

export class FileUploadManager {
  private static instance: FileUploadManager;
  private uploads: Map<string, AbortController> = new Map();

  private constructor() {}

  public static getInstance(): FileUploadManager {
    if (!FileUploadManager.instance) {
      FileUploadManager.instance = new FileUploadManager();
    }
    return FileUploadManager.instance;
  }

  public async uploadFile(
    file: File,
    type: 'image' | 'video' | 'audio',
    onProgress?: (progress: UploadProgress) => void
  ): Promise<{ success: boolean; data?: any; error?: string }> {
    const fileId = Math.random().toString(36).substr(2, 9);
    const controller = new AbortController();
    this.uploads.set(fileId, controller);

    const updateProgress = (progress: number, status: UploadProgress['status'], error?: string) => {
      if (onProgress) {
        onProgress({
          fileId,
          fileName: file.name,
          progress,
          status,
          error,
        });
      }
    };

    try {
      updateProgress(0, 'uploading');
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiClient.getBaseURL()}/api/upload/${type}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiClient.getAuthToken()}`,
        },
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Upload failed');
      }

      const data = await response.json();
      updateProgress(100, 'completed');
      return { success: true, data };
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to upload file';
      updateProgress(0, 'error', errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      this.uploads.delete(fileId);
    }
  }

  public async uploadFiles(
    files: File[],
    type: 'image' | 'video' | 'audio',
    onFileProgress?: (progress: UploadProgress) => void
  ): Promise<{ [key: string]: { success: boolean; data?: any; error?: string } }> {
    const uploadPromises = files.map(file => 
      this.uploadFile(file, type, onFileProgress)
        .then(result => ({ fileId: file.name, result }))
    );

    const results = await Promise.all(uploadPromises);
    return results.reduce((acc, { fileId, result }) => {
      acc[fileId] = result;
      return acc;
    }, {} as { [key: string]: { success: boolean; data?: any; error?: string } });
  }

  public cancelUpload(fileId: string): void {
    const controller = this.uploads.get(fileId);
    if (controller) {
      controller.abort();
      this.uploads.delete(fileId);
    }
  }

  public cancelAllUploads(): void {
    this.uploads.forEach(controller => controller.abort());
    this.uploads.clear();
  }
}

export const fileUploadManager = FileUploadManager.getInstance();
