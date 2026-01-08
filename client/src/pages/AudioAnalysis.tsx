import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Mic, Upload, CheckCircle, AlertCircle, Loader2, Eye, FileText, BarChart3 } from 'lucide-react';
import { apiClient } from '../lib/api';
import logger from '../lib/logger';

export default function AudioAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'live'>('upload');
  const [analysisState, setAnalysisState] = useState<{
    status: 'idle' | 'analyzing' | 'success' | 'error';
    result: any;
    error: string | null;
  }>({
    status: 'idle',
    result: null,
    error: null
  });
  const abortControllerRef = useRef<AbortController | null>(null);

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

  const validateProof = (result: any): boolean => {
    if (!result?.proof) {
      throw new Error('Missing proof of analysis');
    }

    const { proof } = result;
    
    if (typeof proof !== 'object' || proof === null) {
      throw new Error('Invalid proof format');
    }

    if (typeof proof.model_identity !== 'string' || !proof.model_identity) {
      throw new Error('Invalid model identity in proof');
    }

    if (typeof proof.inference_time !== 'number' || proof.inference_time <= 0) {
      throw new Error('Invalid inference time in proof');
    }

    if (proof.authority !== 'SentinelAgent') {
      throw new Error('Invalid analysis authority');
    }

    return true;
  };

  const analyzeAudio = useCallback(async () => {
    if (!selectedFile) return;
    
    // Cancel any in-flight requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setAnalysisState({
      status: 'analyzing',
      result: null,
      error: null
    });

    try {
      logger.info('Starting audio analysis', {
        filename: selectedFile.name,
        size: selectedFile.size
      });

      // Use the correct method from apiClient
      const result = await apiClient.post('/api/analyze/audio', {
        file: selectedFile
      }, {
        signal: controller.signal,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (controller.signal.aborted) return;

      logger.info('Analysis completed', { success: result.data.success });

      if (!result || !result.data) {
        throw new Error('Invalid response from server');
      }

      // Strict proof validation
      validateProof(result.data);

      setAnalysisState({
        status: 'success',
        result: result.data,
        error: null
      });
    } catch (error: any) {
      if (error.name === 'AbortError' || error.message?.includes('aborted')) {
        logger.info('Analysis was cancelled');
        return;
      }

      logger.error('Analysis failed', error);
      
      const errorMessage = error.response?.status === 401 
        ? 'Authentication failed. Please log in again.'
        : 'Analysis failed. Please try again.';

      setAnalysisState(prev => ({
        ...prev,
        status: 'error',
        error: errorMessage
      }));
    } finally {
      abortControllerRef.current = null;
    }
  }, [selectedFile]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return (
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

        {
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
                            analyzeAudio();
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

              <div className="flex items-center gap-2 mt-4 text-sm text-gray-400">
                <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Only audio files are accepted • Maximum file size: 50MB</span>
              </div>
            </div>

            {/* Analysis Results */}
            {analysisState.status === 'error' && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-2 text-red-400">
                  <AlertCircle className="w-5 h-5" />
                  <span>{analysisState.error || 'An error occurred during analysis'}</span>
                </div>
              </div>
            )}

            {analysisState.status === 'success' && analysisState.result && (
              <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${analysisState.result.authenticity === 'AUTHENTIC MEDIA'
                      ? 'bg-green-500/10 border-2 border-green-500/30'
                      : 'bg-red-500/10 border-2 border-red-500/30'
                    }`}>
                    {analysisState.result.authenticity === 'AUTHENTIC MEDIA' ? (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    ) : (
                      <AlertCircle className="w-8 h-8 text-red-500" />
                    )}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-1">
                      {analysisState.result.authenticity === 'AUTHENTIC MEDIA' ? 'Authentic Media' : 'Potential Deepfake'}
                    </h2>
                    <p className="text-gray-400">
                      Confidence: <span className="text-white font-semibold">{analysisState.result.confidence.toFixed(1)}%</span>
                    </p>
                    {analysisState.result.proof && (
                      <div className="mt-2 text-xs text-gray-400">
                        <p>Model: {analysisState.result.proof.model_identity}</p>
                        <p>Inference: {analysisState.result.proof.inference_time.toFixed(2)}s</p>
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
                      {analysisState.result.key_findings?.map((finding: string, index: number) => (
                        <div key={index} className="flex items-start gap-2">
                          <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                          <p className="text-gray-300 text-sm">{finding}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Technical Details */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                      Analysis Scores
                    </h3>
                    <div className="space-y-3">
                      {analysisState.result.detailed_analysis && (
                        <>
                          {analysisState.result.detailed_analysis.voice_analysis && (
                            <div className="flex justify-between items-center">
                              <span className="text-gray-300 text-sm">Voice Consistency</span>
                              <span className="text-white font-semibold">
                                {(analysisState.result.detailed_analysis.voice_analysis.consistency_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                          {analysisState.result.detailed_analysis.spectral_analysis && (
                            <div className="flex justify-between items-center">
                              <span className="text-gray-300 text-sm">Spectral Analysis</span>
                              <span className="text-white font-semibold">
                                {(analysisState.result.detailed_analysis.spectral_analysis.spectral_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                          {analysisState.result.detailed_analysis.background_analysis && (
                            <div className="flex justify-between items-center">
                              <span className="text-gray-300 text-sm">Background Noise</span>
                              <span className="text-white font-semibold">
                                {(analysisState.result.detailed_analysis.background_analysis.noise_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                        </>
                      )}
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 text-sm">Processing Time</span>
                        <span className="text-white font-semibold">
                          {analysisState.result.technical_details?.processing_time_seconds?.toFixed(2) || '0.0'}s
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Case ID and Timestamp */}
                <div className="mt-6 pt-6 border-t border-gray-700">
                  <div className="flex justify-between items-center text-sm text-gray-400">
                    <span>Case ID: {analysisState.result.case_id}</span>
                    <span>Analyzed: {new Date(analysisState.result.analysis_date).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}
          </>
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
  );
};
