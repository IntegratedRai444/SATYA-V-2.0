import React, { useState, useCallback, useEffect } from 'react';
import { Loader2, Upload, Eye, CheckCircle, AlertCircle, FileText, BarChart3, Layers, FileImage, FileVideo, FileAudio } from 'lucide-react';
import { useMultimodalAnalysis } from '../hooks/useApi';
import { pollAnalysisResult } from '../lib/analysis/pollResult';

interface AnalysisResult {
  result: {
    isAuthentic: boolean;
    confidence: number;
    details: {
      isDeepfake: boolean;
      modelInfo: Record<string, unknown>;
    };
    metrics: {
      processingTime: number;
      modelVersion: string;
    };
  };
  key_findings?: string[];
  case_id?: string;
  analysis_date?: string;
  technical_details?: {
    processing_time_seconds?: number;
  };
  id?: string;
  proof?: {
    frames_analyzed?: number;
  };
  requestId?: string;
  createdAt?: string;
}

const MultimodalAnalysis = () => {
  const [dragActive, setDragActive] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'failed'>('idle');
  const [selectedFiles, setSelectedFiles] = useState<{[key: string]: File}>({});
  const [error, setError] = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const { analyzeMultimodal, isAnalyzing } = useMultimodalAnalysis();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files) {
      const files = Array.from(e.dataTransfer.files);
      processFiles(files);
    }
  }, []);

  const processFiles = (files: File[]) => {
    const newFiles: {[key: string]: File} = {};
    
    files.forEach(file => {
      if (file.type.startsWith('image/')) {
        newFiles.image = file;
      } else if (file.type.startsWith('video/')) {
        newFiles.video = file;
      } else if (file.type.startsWith('audio/')) {
        newFiles.audio = file;
      }
    });
    
    setSelectedFiles(newFiles);
    setError('');
    setAnalysisResult(null);
  };

  const handleFileChange = (type: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    let validTypes: string[] = [];
    let maxSize = 0;
    
    switch (type) {
      case 'image':
        validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
        maxSize = 50 * 1024 * 1024; // 50MB
        break;
      case 'video':
        validTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
        maxSize = 500 * 1024 * 1024; // 500MB
        break;
      case 'audio':
        validTypes = ['audio/mp3', 'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/webm'];
        maxSize = 100 * 1024 * 1024; // 100MB
        break;
    }

    if (!validTypes.includes(file.type)) {
      setError(`Invalid ${type} file type. Please upload a valid ${type} file.`);
      return;
    }

    if (file.size > maxSize) {
      setError(`${type.charAt(0).toUpperCase() + type.slice(1)} file is too large. Maximum size is ${maxSize / 1024 / 1024}MB`);
      return;
    }

    setSelectedFiles(prev => ({ ...prev, [type]: file }));
    setError('');
  };

  const handleAnalyze = async () => {
    if (Object.keys(selectedFiles).length === 0) return;
    
    setError('');
    setAnalysisResult(null);
    setAnalysisStatus('uploading');
    
    try {
      const filesArray = Object.values(selectedFiles);
      const response = await analyzeMultimodal({ files: filesArray });
      
      if (!response?.jobId) {
        throw new Error('No jobId returned from backend');
      }

      setJobId(response.jobId);
      setAnalysisStatus('processing');
    } catch (err) {
      if (import.meta.env.DEV) {
        console.error('Analysis failed:', err);
      }
      setError(err instanceof Error ? err.message : 'Failed to analyze files. Please try again.');
      setAnalysisStatus('failed');
    }
  };

  // Poll for results when jobId is available
  useEffect(() => {
    if (jobId && analysisStatus === 'processing') {
      const polling = pollAnalysisResult(jobId, {
        onProgress: () => {
          // Update progress if needed
        }
      });
      
      // Handle polling promise
      polling.promise
        .then((job) => {
          if (job.status === 'completed' && job.result) {
            const analysisResult: AnalysisResult = {
              result: {
                isAuthentic: !job.result.details.isDeepfake,
                confidence: job.result.confidence,
                details: {
                  isDeepfake: job.result.details.isDeepfake,
                  modelInfo: job.result.details.modelInfo,
                },
                metrics: {
                  processingTime: job.result.metrics.processingTime,
                  modelVersion: job.result.metrics.modelVersion
                }
              },
              key_findings: [
                !job.result.details.isDeepfake 
                  ? 'No signs of manipulation detected across modalities' 
                  : 'Potential signs of manipulation found in one or more modalities',
                `Confidence: ${(job.result.confidence * 100).toFixed(1)}%`,
                `Model: ${job.result.details.modelInfo?.name || 'SatyaAI Multimodal'}`,
                'Cross-modal analysis completed successfully'
              ],
              case_id: job.id || `CASE_${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
              analysis_date: new Date().toISOString(),
              technical_details: {
                processing_time_seconds: job.result.metrics.processingTime,
              }
            };
            setAnalysisResult(analysisResult);
            setAnalysisStatus('completed');
          }
        })
        .catch((err: { message?: string }) => {
          setError(err.message || 'Analysis failed');
          setAnalysisStatus('failed');
        });
      
      // Return cleanup function
      return polling.cancel;
    }
  }, [jobId, analysisStatus]);

  const removeFile = (type: string) => {
    setSelectedFiles(prev => {
      const newFiles = { ...prev };
      delete newFiles[type];
      return newFiles;
    });
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-3">
          <div className="w-14 h-14 bg-orange-500/10 rounded-xl flex items-center justify-center border border-orange-500/20">
            <Layers className="w-7 h-7 text-orange-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Multimodal Analysis</h1>
            <p className="text-gray-400 text-sm">Cross-media deepfake detection with 92.5% accuracy</p>
          </div>
        </div>

        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full bg-green-500 animate-pulse`}></div>
            <span className="text-sm text-gray-400">System Ready</span>
          </div>
          <span className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-3 py-1 text-xs font-semibold rounded-md">
            EXPERIMENTAL - Results may not be accurate
          </span>
        </div>

        <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
          <div className="text-center mb-6">
            <h2 className="text-xl font-semibold text-white mb-2">Upload Files for Cross-Modal Analysis</h2>
            <p className="text-gray-400 text-sm">Upload multiple file types for comprehensive analysis</p>
          </div>

          <div
            className={`border-2 border-dashed rounded-xl p-16 transition-all duration-200 ${
              dragActive
                ? 'border-orange-500 bg-orange-500/5'
                : Object.keys(selectedFiles).length > 0
                  ? 'border-green-500/50 bg-green-500/5'
                  : error
                    ? 'border-red-500/50 bg-red-500/5'
                    : 'border-gray-600 bg-[#1e2128] hover:bg-[#252930] hover:border-gray-500/60'
            } cursor-pointer`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('multimodalFileInput')?.click()}
          >
            <input
              id="multimodalFileInput"
              type="file"
              multiple
              accept="image/*,video/*,audio/*"
              onChange={(e) => e.target.files && processFiles(Array.from(e.target.files))}
              className="hidden"
            />

            <div className="flex flex-col items-center gap-4">
              {Object.keys(selectedFiles).length > 0 ? (
                <>
                  <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-10 h-10 text-green-500" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white mb-1">
                      {Object.keys(selectedFiles).length} files selected
                    </p>
                    <p className="text-gray-400 text-sm">
                      {Object.entries(selectedFiles).map(([type, file]) => 
                        `${type.charAt(0).toUpperCase() + type.slice(1)}: ${file.name}`
                      ).join(' | ')}
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAnalyze();
                      }}
                      disabled={isAnalyzing || Object.keys(selectedFiles).length === 0}
                      className="bg-orange-500 hover:bg-orange-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Eye className="w-5 h-5" />
                          Analyze Files
                        </>
                      )}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedFiles({});
                        setAnalysisResult(null);
                        setError('');
                      }}
                      disabled={isAnalyzing}
                      className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                    >
                      <Upload className="w-5 h-5" />
                      Clear Files
                    </button>
                  </div>
                </>
              ) : error ? (
                <>
                  <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center">
                    <AlertCircle className="w-10 h-10 text-red-500" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-red-400 mb-1">Invalid Files</p>
                    <p className="text-gray-400 text-sm">{error}</p>
                  </div>
                  <button className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200">
                    Choose Another File
                  </button>
                </>
              ) : (
                <>
                  <div className="w-20 h-20 bg-orange-500/10 rounded-full flex items-center justify-center">
                    <Layers className="w-10 h-10 text-orange-400" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white mb-1">Drop your files here</p>
                    <p className="text-gray-400 text-sm">Supports Images, Videos, and Audio files</p>
                  </div>
                  <button className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200">
                    Choose Files
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Individual File Upload Areas */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            {/* Image Upload */}
            <div className="bg-[#1e2128] border border-gray-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <FileImage className="w-5 h-5 text-blue-400" />
                <span className="text-white font-medium">Image</span>
              </div>
              {selectedFiles.image ? (
                <div className="text-sm">
                  <p className="text-green-400 truncate">{selectedFiles.image.name}</p>
                  <p className="text-gray-400 text-xs">{formatFileSize(selectedFiles.image.size)}</p>
                  <button
                    onClick={() => removeFile('image')}
                    className="text-red-400 hover:text-red-300 text-xs mt-2"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => document.getElementById('imageInput')?.click()}
                  className="w-full py-2 text-sm bg-blue-600/20 text-blue-400 rounded border border-blue-600/30 hover:bg-blue-600/30 transition-colors"
                >
                  Select Image
                </button>
              )}
              <input
                id="imageInput"
                type="file"
                accept="image/*"
                onChange={handleFileChange('image')}
                className="hidden"
              />
            </div>

            {/* Video Upload */}
            <div className="bg-[#1e2128] border border-gray-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <FileVideo className="w-5 h-5 text-green-400" />
                <span className="text-white font-medium">Video</span>
              </div>
              {selectedFiles.video ? (
                <div className="text-sm">
                  <p className="text-green-400 truncate">{selectedFiles.video.name}</p>
                  <p className="text-gray-400 text-xs">{formatFileSize(selectedFiles.video.size)}</p>
                  <button
                    onClick={() => removeFile('video')}
                    className="text-red-400 hover:text-red-300 text-xs mt-2"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => document.getElementById('videoInput')?.click()}
                  className="w-full py-2 text-sm bg-green-600/20 text-green-400 rounded border border-green-600/30 hover:bg-green-600/30 transition-colors"
                >
                  Select Video
                </button>
              )}
              <input
                id="videoInput"
                type="file"
                accept="video/*"
                onChange={handleFileChange('video')}
                className="hidden"
              />
            </div>

            {/* Audio Upload */}
            <div className="bg-[#1e2128] border border-gray-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <FileAudio className="w-5 h-5 text-purple-400" />
                <span className="text-white font-medium">Audio</span>
              </div>
              {selectedFiles.audio ? (
                <div className="text-sm">
                  <p className="text-green-400 truncate">{selectedFiles.audio.name}</p>
                  <p className="text-gray-400 text-xs">{formatFileSize(selectedFiles.audio.size)}</p>
                  <button
                    onClick={() => removeFile('audio')}
                    className="text-red-400 hover:text-red-300 text-xs mt-2"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => document.getElementById('audioInput')?.click()}
                  className="w-full py-2 text-sm bg-purple-600/20 text-purple-400 rounded border border-purple-600/30 hover:bg-purple-600/30 transition-colors"
                >
                  Select Audio
                </button>
              )}
              <input
                id="audioInput"
                type="file"
                accept="audio/*"
                onChange={handleFileChange('audio')}
                className="hidden"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4 text-sm text-gray-400">
            <svg
              className="w-4 h-4 text-orange-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Upload multiple file types for enhanced cross-modal analysis</span>
          </div>
        </div>

        {/* Results Section */}
        {analysisResult && (
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                analysisResult.result.isAuthentic ? 'bg-green-500/10' : 'bg-red-500/10'
              }`}>
                {analysisResult.result.isAuthentic ? (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                ) : (
                  <AlertCircle className="w-6 h-6 text-red-500" />
                )}
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">
                  {analysisResult.result.isAuthentic ? 'Authentic Media Detected' : 'Potential Manipulation Detected'}
                </h3>
                <p className="text-gray-400 text-sm">
                  Confidence: {(analysisResult.result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-orange-400" />
                  Key Findings
                </h4>
                <ul className="space-y-2">
                  {analysisResult.key_findings?.map((finding, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-2"></div>
                      <span className="text-gray-300 text-sm">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-orange-400" />
                  Technical Details
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Case ID:</span>
                    <span className="text-white">{analysisResult.case_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Processing Time:</span>
                    <span className="text-white">{analysisResult.technical_details?.processing_time_seconds}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Analysis Date:</span>
                    <span className="text-white">
                      {new Date(analysisResult.analysis_date || '').toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <p className="text-yellow-400 text-sm">
                ⚠️ This is an experimental feature. Results may not be accurate and should not be used for critical decisions.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MultimodalAnalysis;
