import { useState, useCallback } from 'react';
import apiClient from '../lib/api';
import logger from '../lib/logger';

type MediaType = 'image' | 'video' | 'audio';

export function useMediaAnalysis(type: MediaType) {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [error, setError] = useState<string>('');

    const analyze = useCallback(async (file: File) => {
        if (!file) return;

        setIsAnalyzing(true);
        setError('');

        try {
            logger.info(`Starting ${type} analysis`, {
                filename: file.name,
                size: file.size,
                authenticated: apiClient.isAuthenticated()
            });

            let result;
            switch (type) {
                case 'image':
                    result = await apiClient.analyzeImage(file);
                    break;
                case 'video':
                    result = await apiClient.analyzeVideo(file);
                    break;
                case 'audio':
                    result = await apiClient.analyzeAudio(file);
                    break;
            }

            logger.info('Analysis completed', { success: result.success });

            // Handle different response formats
            if (result.result) {
                setAnalysisResult(result.result);
            } else if (result.success !== false) {
                setAnalysisResult(result);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error: any) {
            logger.error('Analysis failed', error, {
                message: error.message,
                response: error.response?.data,
                status: error.response?.status
            });

            // Show detailed error message
            if (error.response?.status === 401) {
                setError('Authentication required. Please log in first.');
            } else if (error.response?.status === 429) {
                setError('Rate limit exceeded. Please try again in a minute.');
            } else if (error.message?.includes('Network Error') || error.code === 'ECONNREFUSED') {
                setError('Cannot connect to server. Start all services with: npm run start:all');
            } else if (error.message?.includes('Python') || error.message?.includes('fallback')) {
                setError('Python AI server is not running. Start it with: python server/python/app.py');
            } else {
                setError(error.message || 'Analysis failed. Please try again.');
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
