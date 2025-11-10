import React, { createContext, useContext } from 'react';
import { useBatchProcessing, type BatchFile } from '@/hooks/useBatchProcessing';

interface BatchProcessingContextType {
  files: BatchFile[];
  isProcessing: boolean;
  addFiles: (files: File[]) => void;
  removeFile: (id: string) => void;
  processBatch: () => Promise<BatchFile[]>;
  clearFiles: () => void;
  updateFileStatus?: (id: string, status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error', progress: number, data?: Partial<BatchFile>) => void;
  handleScanUpdate?: (update: any) => void;
}

const BatchProcessingContext = createContext<BatchProcessingContextType | undefined>(undefined);

export const BatchProcessingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const {
    files,
    isProcessing,
    addFiles,
    removeFile,
    processBatch,
    clearFiles,
  } = useBatchProcessing();

  return (
    <BatchProcessingContext.Provider
      value={{
        files,
        isProcessing,
        addFiles,
        removeFile,
        processBatch,
        clearFiles,
      }}
    >
      {children}
    </BatchProcessingContext.Provider>
  );
};

export const useBatchProcessingContext = () => {
  const context = useContext(BatchProcessingContext);
  if (!context) {
    throw new Error('useBatchProcessingContext must be used within a BatchProcessingProvider');
  }
  return context;
};
