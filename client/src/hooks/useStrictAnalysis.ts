import { useState, useCallback, useRef } from 'react';
import { toast } from '@/components/ui/use-toast';
import { validateProof } from '@/lib/utils/proofValidation';

type AnalysisState = 'idle' | 'analyzing' | 'success' | 'error';

interface UseStrictAnalysisOptions<T> {
  analysisFn: (file: File) => Promise<T>;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

export function useStrictAnalysis<T extends { proof?: any }>({
  analysisFn,
  onSuccess,
  onError,
}: UseStrictAnalysisOptions<T>) {
  const [state, setState] = useState<AnalysisState>('idle');
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setState('idle');
    setError(null);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  const analyze = useCallback(
    async (file: File) => {
      // Reset any previous state
      reset();
      
      // Create new abort controller for this request
      abortControllerRef.current = new AbortController();
      
      setState('analyzing');
      setError(null);

      try {
        const result = await analysisFn(file);
        
        // Validate proof before proceeding
        if (!result.proof) {
          throw new Error('Analysis result is missing proof');
        }
        
        const { isValid, error: validationError } = validateProof(result.proof);
        
        if (!isValid) {
          throw new Error(validationError || 'Invalid proof of analysis');
        }
        
        setState('success');
        onSuccess?.(result);
        return result;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Analysis failed');
        console.error('Analysis error:', error);
        
        setState('error');
        setError(error.message);
        onError?.(error);
        
        // Show user-friendly error message
        toast({
          title: 'Analysis Failed',
          description: error.message || 'An error occurred during analysis',
          variant: 'destructive',
        });
        
        throw error;
      } finally {
        abortControllerRef.current = null;
      }
    },
    [analysisFn, onError, onSuccess, reset]
  );

  // Cancel the current analysis
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setState('idle');
      setError(null);
    }
  }, []);

  return {
    state,
    error,
    analyze,
    cancel,
    reset,
    isIdle: state === 'idle',
    isAnalyzing: state === 'analyzing',
    isSuccess: state === 'success',
    isError: state === 'error',
  };
}

// Example usage:
/*
const { state, analyze, cancel } = useStrictAnalysis({
  analysisFn: async (file) => {
    const response = await analysisService.analyzeImage(file);
    return response;
  },
  onSuccess: (result) => {
    // Handle successful analysis with validated proof
    console.log('Analysis result with valid proof:', result);
  },
  onError: (error) => {
    // Handle error (already shown to user via toast)
    console.error('Analysis failed:', error);
  },
});
*/
