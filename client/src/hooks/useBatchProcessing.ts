import { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from '@/components/ui/use-toast';
import { apiClient } from '../lib/api';
import { useRealtime } from './useRealtime';

export interface BatchFile {
  id: string;
  file: File;
  name: string;
  size: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: unknown;
  error?: string;
  scanId?: string;
}

export const useBatchProcessing = () => {
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const { subscribeToScan } = useRealtime(); // Changed hook
  const queryClient = useQueryClient();

  // Add files to batch
  const addFiles = useCallback((newFiles: File[]) => {
    setFiles(prev => [
      ...prev,
      ...newFiles.map(file => ({
        id: `${Date.now()}-${file.name}`,
        file,
        name: file.name,
        size: file.size,
        status: 'pending' as const,
        progress: 0,
      })),
    ]);
  }, []);

  // Remove file from batch
  const removeFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(file => file.id !== id));
  }, []);

  // Clear all files
  const clearFiles = useCallback(() => {
    setFiles([]);
  }, []);

  // Process batch mutation
  const processBatchMutation = useMutation({
    mutationFn: async (filesToProcess: BatchFile[]) => {
      setIsProcessing(true);
      const results: BatchFile[] = [];

      for (const file of filesToProcess) {
        try {
          // Update file status to uploading
          updateFileStatus(file.id, 'uploading', 0);

          // Upload file and handle response
          const formData = new FormData();
          formData.append('file', file.file);

          const response = await apiClient.post<{ success: boolean; data?: { scanId: string }; error?: string }>(
            '/v2/batch/process',
            formData,
            {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
              onUploadProgress: (progressEvent: { loaded: number; total?: number }) => {
                const progress = progressEvent.total
                  ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
                  : 0;
                updateFileStatus(file.id, 'uploading', progress);
              }
            }
          );

          const responseData = response.data;

          if (!responseData.success) {
            throw new Error(responseData.error || 'Failed to upload file');
          }

          const data = responseData.data;
          if (!data || !data.scanId) {
            throw new Error('Invalid response from server: missing scanId');
          }

          // Subscribe to scan updates
          subscribeToScan(data.scanId);

          // Update file status to processing
          updateFileStatus(file.id, 'processing', 100, { result: { scanId: data.scanId } });
          results.push({ ...file, status: 'processing' as const, progress: 100 });
        } catch (error) {
          console.error(`Error processing file ${file.name}: `, error);
          const errorMessage = error instanceof Error ? error.message : 'Failed to process file';
          toast({
            title: `Error processing ${file.name} `,
            description: errorMessage,
            variant: 'destructive',
          });
          updateFileStatus(file.id, 'error', 0, {
            error: errorMessage
          });
        }
      }

      return results;
    },
    onSettled: () => {
      setIsProcessing(false);
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['scans'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
    },
  });

  // Update file status helper
  const updateFileStatus = (
    id: string,
    status: BatchFile['status'],
    progress: number,
    data: Partial<BatchFile> = {}
  ) => {
    setFiles(prev =>
      prev.map(file =>
        file.id === id
          ? { ...file, status, progress, ...data } as BatchFile
          : file
      )
    );
  };

  // Handle WebSocket scan updates
  const handleScanUpdate = useCallback((update: { scanId: string; status: BatchFile['status']; progress?: number; error?: string; result?: unknown }) => {
    setFiles(prev =>
      prev.map(file => {
        if ((file.result as { scanId?: string })?.scanId === update.scanId) {
          return {
            ...file,
            status: update.status,
            progress: update.progress || file.progress,
            result: { ...(file.result as Record<string, unknown>), ...update },
            error: update.error || file.error,
          } as BatchFile;
        }
        return file;
      })
    );
  }, []);

  return {
    files,
    isProcessing,
    addFiles,
    removeFile,
    clearFiles,
    processBatch: () => processBatchMutation.mutateAsync(files.filter(f => f.status === 'pending')),
    updateFileStatus,
    handleScanUpdate,
  };
};

export default useBatchProcessing;
