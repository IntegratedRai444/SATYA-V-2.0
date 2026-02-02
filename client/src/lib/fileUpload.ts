import { buildApiUrl } from './config/urls';

export interface UploadProgress {
  fileId: string;
  fileName: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
  result?: unknown;
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
  ): Promise<{ success: boolean; data?: unknown; error?: string }> {
    const fileId = Math.random().toString(36).substr(2, 9);
    const controller = new AbortController();
    this.uploads.set(fileId, controller);

    const updateProgress = (progress: number, status: UploadProgress['status'], error?: string, result?: unknown) => {
      if (onProgress) {
        onProgress({
          fileId,
          fileName: file.name,
          progress,
          status,
          error,
          result,
        });
      }
    };

    try {
      updateProgress(0, 'uploading');
      
      const token = localStorage.getItem('access_token');
      console.log("File upload auth token:", token ? "Bearer [REDACTED]" : "null");
      
      if (!token) {
        throw new Error("User session missing - cannot upload file");
      }
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(buildApiUrl('/upload/' + type), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errorData: { message?: string } = {};
        try {
          errorData = await response.json();
        } catch (parseError) {
          console.error("Failed to parse error response:", parseError);
        }
        throw new Error(errorData.message || 'Upload failed');
      }

      const data = await response.json();
      updateProgress(100, 'completed', undefined, data);
      return { success: true, data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload file';
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
  ): Promise<{ [key: string]: { success: boolean; data?: unknown; error?: string } }> {
    const uploadPromises = files.map(file => 
      this.uploadFile(file, type, onFileProgress)
        .then(result => ({ fileId: file.name, result }))
    );

    const results = await Promise.all(uploadPromises);
    return results.reduce((acc, { fileId, result }) => {
      acc[fileId] = result;
      return acc;
    }, {} as { [key: string]: { success: boolean; data?: unknown; error?: string } });
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
