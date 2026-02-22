import React, { useState, useCallback, useEffect } from 'react';
import { Loader2, Upload, Mic, CheckCircle, AlertCircle, FileText, BarChart3, Eye } from 'lucide-react';
import { useAudioAnalysis } from '@/hooks/useApi';
import { pollAnalysisResult } from '@/lib/analysis/pollResult';
import { AnalysisResult } from '@/lib/api/services/analysisService';
import type { AnalysisJobStatus } from '@/lib/analysis/pollResult';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';

export default function AudioAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'live'>('upload');
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'failed'>('idle');
  
  // Use the proper hook for audio analysis
  const { analyzeAudio: analyzeAudioHook } = useAudioAnalysis();
  
  const [analysisState, setAnalysisState] = useState<{
    status: 'idle' | 'analyzing' | 'success' | 'error';
    result: AnalysisResult | null;
    error: string | null;
  }>({
    status: 'idle',
    result: null,
    error: null
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const validateFile = (file: File): boolean => {
    const validTypes = ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/mp4', 'audio/webm'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    let errorMessage = '';

    if (!validTypes.includes(file.type)) {
      errorMessage = 'Invalid file type. Please upload an audio file (MP3, WAV, OGG, M4A, MP4, or WebM).';
      setAnalysisState(prev => ({ ...prev, error: errorMessage }));
      return false;
    }

    if (file.size > maxSize) {
      errorMessage = 'File is too large. Maximum size is 50MB.';
      setAnalysisState(prev => ({ ...prev, error: errorMessage }));
      return false;
    }

    return true;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setAnalysisState({
          status: 'idle',
          result: null,
          error: null
        });
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setAnalysisState({
          status: 'idle',
          result: null,
          error: null
        });
      }
    }
  };

  const handleAnalyzeAudio = useCallback(async () => {
    if (!selectedFile) return;

    setAnalysisStatus('uploading');
    setAnalysisState({ status: 'analyzing', result: null, error: null });

    try {
      const response = await analyzeAudioHook({ 
        file: selectedFile, 
        options: { includeDetails: true } 
      });
      
      if (!response?.jobId) {
        throw new Error('No jobId returned from backend');
      }

      setJobId(response.jobId);
      setAnalysisStatus('processing');
    } catch (err) {
      if (import.meta.env.DEV) {
        console.error('Audio analysis failed:', err);
      }
      setAnalysisState({
        status: 'error',
        result: null,
        error: err instanceof Error ? err.message : 'Failed to analyze audio. Please try again.'
      });
      setAnalysisStatus('failed');
    }
  }, [selectedFile, analyzeAudioHook]);

  // Poll for results when jobId is available
  useEffect(() => {
    let cleanup: (() => void) | null = null;
    if (jobId && analysisStatus === 'processing') {
      const pollResult = pollAnalysisResult(jobId, {
        onProgress: () => {
          // Handle progress updates if needed
        },
        onStatusChange: () => {
          // Handle status changes if needed
        }
      });
      
      pollResult.promise
        .then((job: AnalysisJobStatus) => {
          if (job.status === 'completed' && job.result) {
            const analysisResult: AnalysisResult = {
              result: {
                isAuthentic: job.result.isAuthentic,
                confidence: job.result.confidence,
                details: {
                  isDeepfake: job.result.details.isDeepfake,
                  modelInfo: job.result.details.modelInfo || {},
                },
                metrics: {
                  processingTime: job.result.metrics?.processingTime || 0,
                  modelVersion: job.result.metrics?.modelVersion || '1.0.0'
                }
              },
              id: job.id,
              type: 'audio',
              status: 'completed',
              createdAt: new Date().toISOString(), // Use current time since not provided
              updatedAt: new Date().toISOString()
            };
            setAnalysisState({
              status: 'success',
              result: analysisResult,
              error: null
            });
            setAnalysisStatus('completed');
          } else if (job.status === 'failed') {
            setAnalysisState({
              status: 'error',
              result: null,
              error: job.error || 'Analysis failed'
            });
            setAnalysisStatus('failed');
          }
        })
        .catch((err: { message?: string }) => {
          setAnalysisState({
            status: 'error',
            result: null,
            error: err.message || 'Analysis failed'
          });
          setAnalysisStatus('failed');
        });
      
      cleanup = pollResult.cancel;
    }
    
    return () => {
      if (cleanup) cleanup();
    };
  }, [jobId, analysisStatus]);

  return (
    <ErrorBoundary level="page">
      <div className="min-h-screen bg-[#1a1d24] text-white">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Audio Analysis
            </h1>
            <p className="text-gray-400">Detect potential deepfake audio with advanced AI analysis</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-4 mb-6">
          <button
            // ... (rest of the code remains the same)
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${activeTab === 'upload'
                ? 'bg-blue-500 text-white'
                : 'bg-[#2a2e39] text-gray-400 hover:bg-[#323642]'
              }`}
          >
            <div className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              File Upload
            </div>
          </button>
          <button
            onClick={() => setActiveTab('live')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${activeTab === 'live'
                ? 'bg-blue-500 text-white'
                : 'bg-[#2a2e39] text-gray-400 hover:bg-[#323642]'
              }`}
          >
            <div className="flex items-center gap-2">
              <Mic className="w-4 h-4" />
              Live Analysis
            </div>
          </button>
        </div>

        {activeTab === 'upload' && (
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
            <div className="text-center mb-6">
              <h2 className="text-xl font-semibold text-white mb-2">Upload Audio for Analysis</h2>
              <p className="text-gray-400 text-sm">Drag and drop your audio here, or click to browse</p>
            </div>

            <div
              className={`w-full max-w-2xl mx-auto bg-[#2a2e39] border-2 border-dashed rounded-2xl p-8 text-center transition-colors duration-200 ${
                selectedFile
                  ? 'border-green-500/50 bg-green-500/5 cursor-default'
                  : 'border-blue-400/40 bg-[#1e2128] hover:bg-[#252930] hover:border-blue-400/60 cursor-pointer'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('audio-upload')?.click()}
            >
                <input
                  id="audio-upload"
                  type="file"
                  accept="audio/*"
                  onChange={handleFileChange}
                  className="hidden"
                />

                <div className="flex flex-col items-center gap-4">
                  {selectedFile ? (
                    <>
                      <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-10 h-10 text-green-500" />
                      </div>
                      <div>
                        <p className="text-lg font-medium text-white mb-1">{selectedFile.name}</p>
                        <p className="text-gray-400 text-sm">{formatFileSize(selectedFile.size)}</p>
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAnalyzeAudio();
                          }}
                          disabled={analysisState.status === 'analyzing'}
                          className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                        >
                          {analysisState.status === 'analyzing' ? (
                            <>
                              <Loader2 className="w-5 h-5 animate-spin" />
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <Eye className="w-5 h-5" />
                              Analyze Audio
                            </>
                          )}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFile(null);
                            setAnalysisState({
                              status: 'idle',
                              result: null,
                              error: null
                            });
                          }}
                          disabled={analysisState.status === 'analyzing'}
                          className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                        >
                          <Upload className="w-5 h-5" />
                          Change File
                        </button>
                      </div>
                    </>
                  ) : analysisState.error ? (
                    <>
                      <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center">
                        <AlertCircle className="w-10 h-10 text-red-500" />
                      </div>
                      <div>
                        <p className="text-lg font-medium text-red-400 mb-1">Invalid File</p>
                        <p className="text-gray-400 text-sm">{analysisState.error}</p>
                      </div>
                      <button
                        onClick={() => {
                          setAnalysisState({
                            status: 'idle',
                            result: null,
                            error: null
                          });
                        }}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200"
                      >
                        Choose Another File
                      </button>
                    </>
                  ) : (
                    <>
                      <div className="space-y-2">
                        <div className="flex items-center justify-center gap-2">
                          <Upload className="w-5 h-5 text-blue-400" />
                          <span className="text-blue-400 font-medium">Click to upload</span>
                          <span className="text-gray-400">or drag and drop</span>
                        </div>
                        <p className="text-sm text-gray-400">
                          WAV, MP3, OGG, M4A, MP4, or WebM (max 50MB)
                        </p>
                      </div>
                      <button
                        onClick={() => document.getElementById('audio-upload')?.click()}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200"
                      >
                        Choose File
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>

            )}
            {analysisState.status === 'error' && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-2 text-red-400">
                  <AlertCircle className="w-5 h-5" />
                  <span>{analysisState.error || 'An error occurred during analysis'}</span>
                </div>
              </div>
            )}

            {/* Analysis Results */}
            {analysisState.status === 'success' && analysisState.result && (
              <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${analysisState.result?.result?.isAuthentic
                      ? 'bg-green-500/10 border-2 border-green-500/30'
                      : 'bg-red-500/10 border-2 border-red-500/30'
                    }`}>
                    {analysisState.result?.result?.isAuthentic ? (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    ) : (
                      <AlertCircle className="w-8 h-8 text-red-500" />
                    )}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-1">
                      {analysisState.result?.result?.isAuthentic ? 'Authentic Media' : 'Potential Deepfake'}
                    </h2>
                    <p className="text-gray-400">
                      Confidence: <span className="text-white font-semibold">{((analysisState.result?.result?.confidence || 0) * 100).toFixed(1)}%</span>
                    </p>
                    {analysisState.result?.proof && (
                      <div className="mt-2 text-xs text-gray-400">
                        <p>Model: {analysisState.result.proof.model_name || 'Unknown'}</p>
                        <p>Inference: {analysisState.result.proof.inference_duration?.toFixed(2) || '0.0'}s</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Key Findings */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-400" />
                      Key Findings
                    </h3>
                    <div className="space-y-2">
                      {analysisState.result?.result?.details ? (
                        <div className="space-y-2">
                          {Object.entries(analysisState.result.result.details as Record<string, unknown>).map(([key, value]) => (
                            <div key={key} className="flex items-start gap-2">
                              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                              <p className="text-gray-300 text-sm">
                                <span className="font-semibold">{key}:</span> {String(value)}
                              </p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-gray-400 text-sm">No detailed analysis available</div>
                      )}
                    </div>
                  </div>

                  {/* Technical Details */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                      Analysis Scores
                    </h3>
                    <div className="space-y-3">
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300 text-sm">Processing Time</span>
                          <span className="text-white font-semibold">
                            {analysisState.result?.result?.metrics?.processingTime?.toFixed(2) || '0.0'}s
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300 text-sm">Model Version</span>
                          <span className="text-white font-semibold">
                            {analysisState.result?.result?.metrics?.modelVersion || 'Unknown'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Case ID and Timestamp */}
                <div className="mt-6 pt-6 border-t border-gray-700">
                  <div className="flex justify-between items-center text-sm text-gray-400">
                    <span>Case ID: {analysisState.result?.id}</span>
                    <span>Analyzed: {new Date(analysisState.result?.createdAt || Date.now()).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}

        {/* Analysis Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-blue-400/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Quick Analysis</h3>
            <p className="text-gray-400 text-sm mb-4">Fast detection using core models</p>
            <div className="text-3xl font-bold text-blue-400">~1min</div>
          </div>

          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-blue-400/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Comprehensive</h3>
            <p className="text-gray-400 text-sm mb-4">Deep analysis with all models</p>
            <div className="text-3xl font-bold text-blue-400">~3min</div>
          </div>

          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-blue-400/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Forensic Grade</h3>
            <p className="text-gray-400 text-sm mb-4">Maximum accuracy analysis</p>
            <div className="text-3xl font-bold text-blue-400">~5min</div>
          </div>
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
}
