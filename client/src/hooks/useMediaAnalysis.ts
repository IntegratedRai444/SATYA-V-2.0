import { useState, useCallback } from 'react';
import analysisService from '@/lib/api/services/analysisService';
import logger from '../lib/logger';
import type { AnalysisResult } from '@/lib/api/services/analysisService';

type MediaType = 'image' | 'video' | 'audio';

export function useMediaAnalysis(type: MediaType) {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string>('');

    const analyze = useCallback(async (file: File) => {
        if (!file) return;

        setIsAnalyzing(true);
        setError('');

        try {
            logger.info(`Starting ${type} analysis`, {
                filename: file.name,
                size: file.size,
                authenticated: true // Assuming user is authenticated if they can access this hook
            });

            let jobId: string;
            switch (type) {
                case 'image':
                    jobId = await analysisService.analyzeImage(file);
                    break;
                case 'video':
                    jobId = await analysisService.analyzeVideo(file);
                    break;
                case 'audio':
                    jobId = await analysisService.analyzeAudio(file);
                    break;
                default:
                    throw new Error(`Unsupported media type: ${type}`);
            }

            logger.info('Analysis started', { success: true, jobId });

            // Poll for results
            const pollForResults = async (): Promise<AnalysisResult> => {
                const maxAttempts = 60; // 5 minutes with 5-second intervals
                let attempts = 0;
                
                while (attempts < maxAttempts) {
                    const result = await analysisService.getAnalysisResult(jobId);
                    
                    if (result.status === 'completed') {
                        return result;
                    } else if (result.status === 'failed') {
                        throw new Error(result.error || 'Analysis failed');
                    }
                    
                    // Wait before next poll
                    await new Promise(resolve => setTimeout(resolve, 5000));
                    attempts++;
                }
                
                throw new Error('Analysis timeout');
            };

            const result = await pollForResults();

            // Handle the response format from analysisService
            if (result.status === 'completed' && result.result) {
                setAnalysisResult(result);
            } else if (result.status === 'failed') {
                throw new Error(result.error || 'Analysis failed');
            } else {
                setAnalysisResult(result);
            }
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            const response = error instanceof Error && 'response' in error ? (error as Error & { response?: { data?: unknown; status?: number } }).response : undefined;
            logger.error('Analysis failed', error instanceof Error ? error : new Error(String(error)), {
                message: errorMessage,
                response: response?.data,
                status: response?.status
            });

            // Show detailed error message
            if (response?.status === 401) {
                setError('Authentication required. Please log in first.');
            } else if (response?.status === 429) {
                setError('Rate limit exceeded. Please try again in a minute.');
            } else if (errorMessage.includes('Network Error') || (error as { code?: string }).code === 'ECONNREFUSED') {
                setError('Cannot connect to server. Start all services with: npm run start:all');
            } else if (errorMessage.includes('Python') || errorMessage.includes('fallback')) {
                setError('Python AI server is not running. Start it with: python server/python/app.py');
            } else {
                setError(errorMessage || 'Analysis failed. Please try again.');
            }
        } finally {
            setIsAnalyzing(false);
        }
    }, [type]);

    const resetAnalysis = useCallback(() => {
        setAnalysisResult(null);
        setError('');
    }, []);

    return {
        isAnalyzing,
        analysisResult,
        error,
        analyze,
        resetAnalysis,
        setError,
    };
}
