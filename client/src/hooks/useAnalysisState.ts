/**
 * Shared analysis state hook for consistent UX across all analysis pages
 */

import { useState, useCallback } from 'react';

export type AnalysisStatus = 'idle' | 'validating' | 'uploading' | 'processing' | 'completed' | 'error';

export interface AnalysisState {
  status: AnalysisStatus;
  progress?: number;
  message?: string;
  error?: {
    code: string;
    message: string;
    retryable: boolean;
  };
}

export interface UseAnalysisStateReturn {
  state: AnalysisState;
  setStatus: (status: AnalysisStatus, message?: string) => void;
  setProgress: (progress: number) => void;
  setError: (code: string, message: string, retryable?: boolean) => void;
  reset: () => void;
  isProcessing: boolean;
  isComplete: boolean;
  hasError: boolean;
}

export const useAnalysisState = (): UseAnalysisStateReturn => {
  const [state, setState] = useState<AnalysisState>({
    status: 'idle'
  });

  const setStatus = useCallback((status: AnalysisStatus, message?: string) => {
    setState(prev => ({
      ...prev,
      status,
      message,
      error: undefined
    }));
  }, []);

  const setProgress = useCallback((progress: number) => {
    setState(prev => ({
      ...prev,
      progress: Math.max(0, Math.min(100, progress))
    }));
  }, []);

  const setError = useCallback((code: string, message: string, retryable = false) => {
    setState(prev => ({
      ...prev,
      status: 'error',
      error: { code, message, retryable }
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      status: 'idle'
    });
  }, []);

  return {
    state,
    setStatus,
    setProgress,
    setError,
    reset,
    isProcessing: state.status === 'uploading' || state.status === 'processing',
    isComplete: state.status === 'completed',
    hasError: state.status === 'error'
  };
};
