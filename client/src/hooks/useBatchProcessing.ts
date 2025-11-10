import { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from '@/components/ui/use-toast';
import api from '@/lib/api-client';
import { useWebSocket } from '@/contexts/WebSocketContext';

export interface BatchFile {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: any;
  error?: string;
}

export const useBatchProcessing = () => {
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const { subscribeToScan } = useWebSocket();
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
        type: file.type.split('/')[0],
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
  const processBatch = useMutation(
    async (filesToProcess: BatchFile[]) => {
      setIsProcessing(true);
      const results: BatchFile[] = [];

      for (const file of filesToProcess) {
        try {
          // Update file status to uploading
          updateFileStatus(file.id, 'uploading', 0);

          // Upload file and handle response
          const response = await api.uploadFile<{ scanId: string }>(
            '/v2/batch/process',
            file.file,
            'file',
            (progress) => {
              updateFileStatus(file.id, 'uploading', progress);
            }
          );

          if (!response.success) {
            throw new Error(response.error || 'Failed to upload file');
          }
          
          const data = response.data;
          if (!data || !data.scanId) {
            throw new Error('Invalid response from server: missing scanId');
          }

          // Subscribe to scan updates
          subscribeToScan(data.scanId);

          // Update file status to processing
          updateFileStatus(file.id, 'processing', 100, { result: { scanId: data.scanId } });
          results.push({ ...file, status: 'processing' as const, progress: 100 });
        } catch (error) {
          console.error(`Error processing file ${file.name}:`, error);
          const errorMessage = error instanceof Error ? error.message : 'Failed to process file';
          toast({
            title: `Error processing ${file.name}`,
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
    {
      onSettled: () => {
        setIsProcessing(false);
        // Invalidate queries to refresh data
        queryClient.invalidateQueries(['scans']);
        queryClient.invalidateQueries(['analytics']);
      },
    }
  );

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
  const handleScanUpdate = useCallback((update: any) => {
    setFiles(prev => 
      prev.map(file => {
        if (file.result?.scanId === update.scanId) {
          return {
            ...file,
            status: update.status,
            progress: update.progress || file.progress,
            result: { ...file.result, ...update },
            error: update.error || file.error,
          };
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
    processBatch: () => processBatch.mutateAsync(files.filter(f => f.status === 'pending')),
    updateFileStatus,
    handleScanUpdate,
  };
};

export default useBatchProcessing;
