import React, { useState, useCallback, useRef, useEffect } from 'react';
import logger from '../lib/logger';
import {
  Upload,
  FileImage,
  FileVideo,
  FileAudio,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2,
  Eye,
  Shield,
  Brain,
  Layers
} from 'lucide-react';
import { useMultimodalAnalysis } from '../hooks/useApi';
import { AnalysisResult } from '../hooks/useApi';
import { pollAnalysisResult } from '../lib/analysis/pollResult';
import type { AnalysisJobStatus } from '../lib/analysis/pollResult';

interface AnalysisStateItem {
  id: string;
  filename: string;
  type: 'image' | 'video' | 'audio' | 'multimodal';
  status: 'analyzing' | 'completed' | 'error';
  result?: AnalysisResult;
  error?: string;
}

const MultimodalAnalysis: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'single' | 'multimodal'>('single');
  const [selectedType, setSelectedType] = useState<'image' | 'video' | 'audio'>('image');
  const [results, setResults] = useState<AnalysisStateItem[]>([]);
  const [jobIds, setJobIds] = useState<Record<string, string>>({});
  
  // Use the proper hook for multimodal analysis
  const { analyzeMultimodal, isAnalyzing } = useMultimodalAnalysis();

  // Poll for results when jobIds are available
  useEffect(() => {
    Object.entries(jobIds).forEach(([resultId, jobId]) => {
      const result = results.find(r => r.id === resultId);
      if (result && result.status === 'analyzing') {
        const polling = pollAnalysisResult(jobId, {
          onProgress: (progress: number) => {
            console.log(`Multimodal analysis progress: ${progress}%`);
          }
        });
        
        polling.promise
          .then((job: AnalysisJobStatus) => {
            if (job.status === 'completed' && job.result) {
              setResults(prev => prev.map(r =>
                r.id === resultId
                  ? { 
                      ...r, 
                      status: 'completed', 
                      result: {
                        id: job.id,
                        type: 'multimodal',
                        status: 'completed',
                        result: {
                          isAuthentic: job.result?.isAuthentic ?? false,
                          confidence: job.result?.confidence ?? 0,
                          details: job.result?.details ?? {},
                          metrics: job.result?.metrics ?? { processingTime: 0, modelVersion: '1.0.0' }
                        },
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                        fileName: result.filename,
                        fileSize: 0 // TODO: Calculate actual file size
                      } as AnalysisResult
                    }
                  : r
              ));
              
              // Clean up job ID after completion
              setJobIds(prev => {
                const newJobIds = { ...prev };
                delete newJobIds[resultId];
                return newJobIds;
              });
            }
          })
          .catch((err: { message?: string }) => {
            setResults(prev => prev.map(r =>
              r.id === resultId
                ? { ...r, status: 'error', error: err.message || 'Analysis failed' }
                : r
            ));
          });
        
        return polling.cancel;
      }
    });
  }, [jobIds, results]);
  
  // Webcam feature temporarily disabled

  // File upload handlers
  const handleFileUpload = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    const resultId = `multimodal-${Date.now()}`;
    
    const result: AnalysisStateItem = {
      id: resultId,
      filename: `Multimodal Analysis (${files.length} files)`,
      type: 'multimodal',
      status: 'analyzing'
    };

    setResults(prev => [...prev, result]);

    try {
      // Convert FileList to array for the hook
      const filesArray = Array.from(files);
      
      // Use the hook for analysis
      const response = await analyzeMultimodal({ files: filesArray });

      // Store job ID for polling
      if (response?.jobId) {
        setJobIds(prev => ({ ...prev, [resultId]: response.jobId }));
      }
      
      logger.info('Multimodal analysis started successfully', {
        jobId: response.jobId
      });
      
    } catch (error: unknown) {
      const errorMessage = error instanceof Error 
        ? error.message 
        : 'Multimodal analysis failed. Please try again.';
      
      logger.error('Multimodal analysis failed: ' + errorMessage);
      setResults(prev => prev.map(r =>
        r.id === resultId
          ? { ...r, status: 'error', error: errorMessage }
          : r
      ));
    }
  }, [analyzeMultimodal]);

  // Multimodal analysis
  const handleMultimodalUpload = useCallback(async (files: { [key: string]: File }) => {
    const resultId = `multimodal-${Date.now()}`;

    const result: AnalysisStateItem = {
      id: resultId,
      filename: `Multimodal Analysis (${Object.keys(files).length} files)`,
      type: 'multimodal',
      status: 'analyzing'
    };

    setResults(prev => [...prev, result]);

    try {
      // Convert object to array for the hook
      const filesArray = Object.values(files);
      
      // Use the hook for analysis
      const response = await analyzeMultimodal({ files: filesArray });

      // Store job ID for polling
      if (response?.jobId) {
        setJobIds(prev => ({ ...prev, [resultId]: response.jobId }));
      }
      
      logger.info('Multimodal analysis started successfully', {
        jobId: response.jobId
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setResults(prev => prev.map(r =>
        r.id === resultId
          ? { ...r, status: 'error', error: errorMessage }
          : r
      ));
    }
  }, [analyzeMultimodal]);

  // Webcam functions temporarily disabled

  const getResultIcon = (result: AnalysisStateItem) => {
    if (result.status === 'analyzing') return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
    if (result.status === 'error') return <AlertCircle className="w-5 h-5 text-red-500" />;
    if (result.result?.result?.isAuthentic) return <CheckCircle className="w-5 h-5 text-green-500" />;
    return <AlertCircle className="w-5 h-5 text-red-500" />;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-500';
    if (confidence >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Multimodal Analysis
              </h1>
              <span className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-3 py-1 text-xs font-semibold rounded-md">
                EXPERIMENTAL
              </span>
            </div>
          </div>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Advanced AI-powered deepfake detection with single-file, multimodal, and real-time analysis capabilities
          </p>
          <p className="text-yellow-400/60 text-sm mt-2">
            This feature is experimental and may not provide accurate results. Use with caution.
          </p>
        </div>

        {/* Mode Selection */}
        <div className="flex justify-center mb-8">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-2 flex gap-2">
            <button
              onClick={() => setActiveMode('single')}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${activeMode === 'single'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Upload className="w-4 h-4" />
              Single File
            </button>
            <button
              onClick={() => setActiveMode('multimodal')}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${activeMode === 'multimodal'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Layers className="w-4 h-4" />
              Multimodal
            </button>
            {/* Webcam feature temporarily disabled */}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6">
              {activeMode === 'single' && (
                <SingleFileUpload
                  selectedType={selectedType}
                  setSelectedType={setSelectedType}
                  onFileUpload={handleFileUpload}
                  isAnalyzing={isAnalyzing}
                />
              )}

              {activeMode === 'multimodal' && (
                <MultimodalUpload
                  onMultimodalUpload={handleMultimodalUpload}
                  isAnalyzing={isAnalyzing}
                />
              )}

              {/* Webcam feature temporarily disabled */}
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6">
              <div className="flex items-center gap-2 mb-6">
                <Shield className="w-5 h-5 text-blue-400" />
                <h3 className="text-xl font-semibold text-white">Analysis Results</h3>
              </div>

              {results.length === 0 ? (
                <div className="text-center py-12">
                  <Eye className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No analyses yet</p>
                  <p className="text-gray-500 text-sm mt-2">Upload files to start</p>
                </div>
              ) : (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {results.map((result) => (
                    <div key={result.id} className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
                      <div className="flex items-start gap-3">
                        {getResultIcon(result)}
                        <div className="flex-1 min-w-0">
                          <p className="text-white font-medium truncate">{result.filename}</p>
                          <p className="text-gray-400 text-sm capitalize">{result.type} analysis</p>

                          {result.status === 'completed' && result.result && (
                            <div className="mt-2">
                              <div className="flex items-center gap-2 mb-1">
                                <span className={`text-sm font-medium ${result.result.result?.isAuthentic ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                  {result.result.result?.isAuthentic ? 'AUTHENTIC MEDIA' : 'MANIPULATED MEDIA'}
                                </span>
                                <span className={`text-sm ${getConfidenceColor(result.result.result?.confidence || 0)}`}>
                                  {result.result.result?.confidence}%
                                </span>
                              </div>
                              {result.result.result?.details && (
                                <p className="text-xs text-gray-400 truncate">
                                  Analysis complete
                                </p>
                              )}
                            </div>
                          )}

                          {result.status === 'error' && (
                            <p className="text-red-400 text-sm mt-1">{result.error}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Single File Upload Component
const SingleFileUpload: React.FC<{
  selectedType: 'image' | 'video' | 'audio';
  setSelectedType: (type: 'image' | 'video' | 'audio') => void;
  onFileUpload: (files: FileList | null, type: string) => void;
  isAnalyzing: boolean;
}> = ({ selectedType, setSelectedType, onFileUpload, isAnalyzing }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const getAcceptedTypes = () => {
    switch (selectedType) {
      case 'image': return 'image/*';
      case 'video': return 'video/*';
      case 'audio': return 'audio/*';
      default: return '*/*';
    }
  };

  const getIcon = () => {
    switch (selectedType) {
      case 'image': return <FileImage className="w-8 h-8 text-blue-400" />;
      case 'video': return <FileVideo className="w-8 h-8 text-green-400" />;
      case 'audio': return <FileAudio className="w-8 h-8 text-purple-400" />;
    }
  };

  return (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">Single File Analysis</h3>

      {/* Type Selection */}
      <div className="flex gap-2 mb-6">
        {(['image', 'video', 'audio'] as const).map((type) => (
          <button
            key={type}
            onClick={() => setSelectedType(type)}
            className={`px-4 py-2 rounded-lg font-medium transition-all capitalize ${selectedType === type
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
              }`}
          >{type}
          </button>
        ))}
      </div>

      {/* Upload Area */}
      <div
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
      >
        {getIcon()}
        <p className="text-white font-medium mt-4">Click to upload {selectedType}</p>
        <p className="text-gray-400 text-sm mt-2">Drag and drop files here</p>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={getAcceptedTypes()}
        multiple
        className="hidden"
        onChange={(e) => onFileUpload(e.target.files, selectedType)}
        disabled={isAnalyzing}
      />
    </div>
  );
};

// Multimodal Upload Component
const MultimodalUpload: React.FC<{
  onMultimodalUpload: (files: { [key: string]: File }) => void;
  isAnalyzing: boolean;
}> = ({ onMultimodalUpload, isAnalyzing }) => {
  const [files, setFiles] = useState<{ [key: string]: File }>({});
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (type: string, file: File | null) => {
    if (file) {
      setFiles(prev => ({ ...prev, [type]: file }));
    } else {
      setFiles(prev => {
        const newFiles = { ...prev };
        delete newFiles[type];
        return newFiles;
      });
    }
  };

  const handleAnalyze = () => {
    if (Object.keys(files).length > 0) {
      onMultimodalUpload(files);
      setFiles({});
    }
  };

  return (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">Multimodal Analysis</h3>
      <p className="text-gray-400 mb-6">Upload multiple file types for comprehensive cross-validation analysis</p>

      {/* File Selection Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Image Upload */}
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
          <div className="flex items-center gap-2 mb-3">
            <FileImage className="w-5 h-5 text-blue-400" />
            <span className="text-white font-medium">Image</span>
          </div>
          {files.image ? (
            <div className="text-sm">
              <p className="text-green-400 truncate">{files.image.name}</p>
              <button
                onClick={() => handleFileSelect('image', null)}
                className="text-red-400 hover:text-red-300 text-xs mt-1"
              >
                Remove
              </button>
            </div>
          ) : (
            <button
              onClick={() => imageInputRef.current?.click()}
              className="w-full py-2 text-sm bg-blue-600/20 text-blue-400 rounded border border-blue-600/30 hover:bg-blue-600/30 transition-colors"
            >
              Select Image
            </button>
          )}
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFileSelect('image', e.target.files?.[0] || null)}
          />
        </div>

        {/* Video Upload */}
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
          <div className="flex items-center gap-2 mb-3">
            <FileVideo className="w-5 h-5 text-green-400" />
            <span className="text-white font-medium">Video</span>
          </div>
          {files.video ? (
            <div className="text-sm">
              <p className="text-green-400 truncate">{files.video.name}</p>
              <button
                onClick={() => handleFileSelect('video', null)}
                className="text-red-400 hover:text-red-300 text-xs mt-1"
              >
                Remove
              </button>
            </div>
          ) : (
            <button
              onClick={() => videoInputRef.current?.click()}
              className="w-full py-2 text-sm bg-green-600/20 text-green-400 rounded border border-green-600/30 hover:bg-green-600/30 transition-colors"
            >
              Select Video
            </button>
          )}
          <input
            ref={videoInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => handleFileSelect('video', e.target.files?.[0] || null)}
          />
        </div>

        {/* Audio Upload */}
        <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
          <div className="flex items-center gap-2 mb-3">
            <FileAudio className="w-5 h-5 text-purple-400" />
            <span className="text-white font-medium">Audio</span>
          </div>
          {files.audio ? (
            <div className="text-sm">
              <p className="text-green-400 truncate">{files.audio.name}</p>
              <button
                onClick={() => handleFileSelect('audio', null)}
                className="text-red-400 hover:text-red-300 text-xs mt-1"
              >
                Remove
              </button>
            </div>
          ) : (
            <button
              onClick={() => audioInputRef.current?.click()}
              className="w-full py-2 text-sm bg-purple-600/20 text-purple-400 rounded border border-purple-600/30 hover:bg-purple-600/30 transition-colors"
            >
              Select Audio
            </button>
          )}
          <input
            ref={audioInputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => handleFileSelect('audio', e.target.files?.[0] || null)}
          />
        </div>
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={Object.keys(files).length === 0 || isAnalyzing}
        className="w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-medium rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-purple-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2"
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Zap className="w-4 h-4" />
            Analyze ({Object.keys(files).length} files)
          </>
        )}
      </button>
    </div>
  );
};

// Webcam Capture Component temporarily disabled

export default MultimodalAnalysis;
