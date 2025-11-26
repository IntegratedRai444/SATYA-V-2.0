import React, { useState, useCallback, useRef } from 'react';
import logger from '../lib/logger';
import {
  Upload,
  Camera,
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
import apiClient, { AnalysisResult } from '@/lib/api';

interface AnalysisStateItem {
  id: string;
  filename: string;
  type: 'image' | 'video' | 'audio' | 'multimodal';
  status: 'analyzing' | 'completed' | 'error';
  result?: AnalysisResult;
  error?: string;
}

const SmartAnalysis: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'single' | 'multimodal' | 'webcam'>('single');
  const [selectedType, setSelectedType] = useState<'image' | 'video' | 'audio'>('image');
  const [results, setResults] = useState<AnalysisStateItem[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // File upload handlers
  const handleFileUpload = useCallback(async (files: FileList | null, type: string) => {
    if (!files || files.length === 0) return;

    setIsAnalyzing(true);
    const newResults: AnalysisStateItem[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const resultId = `${Date.now()}-${i}`;

      const result: AnalysisStateItem = {
        id: resultId,
        filename: file.name,
        type: type as any,
        status: 'analyzing'
      };

      newResults.push(result);
      setResults(prev => [...prev, result]);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const endpoint = `/api/analysis/${type}`;
        const response = await apiClient.client.post<AnalysisResult>(endpoint, formData);

        setResults(prev => prev.map(r =>
          r.id === resultId
            ? { ...r, status: 'completed', result: response.data }
            : r
        ));
      } catch (error: any) {
        setResults(prev => prev.map(r =>
          r.id === resultId
            ? { ...r, status: 'error', error: error.message }
            : r
        ));
      }
    }

    setIsAnalyzing(false);
  }, []);

  // Multimodal analysis
  const handleMultimodalUpload = useCallback(async (files: { [key: string]: File }) => {
    setIsAnalyzing(true);
    const resultId = `multimodal-${Date.now()}`;

    const result: AnalysisStateItem = {
      id: resultId,
      filename: `Multimodal Analysis (${Object.keys(files).length} files)`,
      type: 'multimodal',
      status: 'analyzing'
    };

    setResults(prev => [...prev, result]);

    try {
      const formData = new FormData();
      Object.entries(files).forEach(([key, file]) => {
        formData.append(key, file);
      });

      const response = await apiClient.client.post<AnalysisResult>('/api/analysis/multimodal', formData);

      setResults(prev => prev.map(r =>
        r.id === resultId
          ? { ...r, status: 'completed', result: response.data }
          : r
      ));
    } catch (error: any) {
      setResults(prev => prev.map(r =>
        r.id === resultId
          ? { ...r, status: 'error', error: error.message }
          : r
      ));
    }

    setIsAnalyzing(false);
  }, []);

  // Webcam functions
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setWebcamActive(true);
      }
    } catch (error) {
      logger.error('Error accessing webcam', error as Error);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setWebcamActive(false);
    }
  }, []);

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      setIsAnalyzing(true);
      const resultId = `webcam-${Date.now()}`;

      const result: AnalysisStateItem = {
        id: resultId,
        filename: 'Webcam Capture',
        type: 'image',
        status: 'analyzing'
      };

      setResults(prev => [...prev, result]);

      try {
        const formData = new FormData();
        formData.append('file', blob, 'webcam-capture.jpg');

        const response = await apiClient.client.post<AnalysisResult>('/api/analysis/webcam', formData);

        setResults(prev => prev.map(r =>
          r.id === resultId
            ? { ...r, status: 'completed', result: response.data }
            : r
        ));
      } catch (error: any) {
        setResults(prev => prev.map(r =>
          r.id === resultId
            ? { ...r, status: 'error', error: error.message }
            : r
        ));
      }

      setIsAnalyzing(false);
    }, 'image/jpeg', 0.8);
  }, []);

  const getResultIcon = (result: AnalysisStateItem) => {
    if (result.status === 'analyzing') return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
    if (result.status === 'error') return <AlertCircle className="w-5 h-5 text-red-500" />;
    if (result.result?.result?.authenticity === 'AUTHENTIC MEDIA') return <CheckCircle className="w-5 h-5 text-green-500" />;
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
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Smart Analysis
            </h1>
          </div>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Advanced AI-powered deepfake detection with single-file, multimodal, and real-time analysis capabilities
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
            <button
              onClick={() => setActiveMode('webcam')}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${activeMode === 'webcam'
                ? 'bg-pink-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Camera className="w-4 h-4" />
              Live Webcam
            </button>
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

              {activeMode === 'webcam' && (
                <WebcamCapture
                  videoRef={videoRef}
                  canvasRef={canvasRef}
                  webcamActive={webcamActive}
                  onStartWebcam={startWebcam}
                  onStopWebcam={stopWebcam}
                  onCapture={captureFrame}
                  isAnalyzing={isAnalyzing}
                />
              )}
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
                  <p className="text-gray-500 text-sm mt-2">Upload files or use webcam to start</p>
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
                                <span className={`text-sm font-medium ${result.result.result?.authenticity === 'AUTHENTIC MEDIA' ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                  {result.result.result?.authenticity}
                                </span>
                                <span className={`text-sm ${getConfidenceColor(result.result.result?.confidence || 0)}`}>
                                  {result.result.result?.confidence}%
                                </span>
                              </div>
                              {result.result.result?.keyFindings && result.result.result.keyFindings.length > 0 && (
                                <p className="text-xs text-gray-400 truncate">
                                  {result.result.result.keyFindings[0] || 'Analysis complete'}
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

// Webcam Capture Component
const WebcamCapture: React.FC<{
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  webcamActive: boolean;
  onStartWebcam: () => void;
  onStopWebcam: () => void;
  onCapture: () => void;
  isAnalyzing: boolean;
}> = ({ videoRef, canvasRef, webcamActive, onStartWebcam, onStopWebcam, onCapture, isAnalyzing }) => {
  return (
    <div>
      <h3 className="text-xl font-semibold text-white mb-4">Live Webcam Analysis</h3>
      <p className="text-gray-400 mb-6">Real-time deepfake detection using your camera</p>

      {/* Video Display */}
      <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-64 object-cover"
          style={{ display: webcamActive ? 'block' : 'none' }}
        />
        {!webcamActive && (
          <div className="w-full h-64 flex items-center justify-center">
            <div className="text-center">
              <Camera className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">Camera not active</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-3">
        {!webcamActive ? (
          <button
            onClick={onStartWebcam}
            className="flex-1 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center gap-2"
          >
            <Camera className="w-4 h-4" />
            Start Camera
          </button>
        ) : (
          <>
            <button
              onClick={onStopWebcam}
              className="flex-1 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors"
            >
              Stop Camera
            </button>
            <button
              onClick={onCapture}
              disabled={isAnalyzing}
              className="flex-1 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Capture & Analyze
                </>
              )}
            </button>
          </>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default SmartAnalysis;
