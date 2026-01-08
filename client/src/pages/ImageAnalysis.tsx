import { useState, useCallback } from 'react';
import { Loader2, Upload, Eye, CheckCircle, AlertCircle, FileText, BarChart3, Camera } from 'lucide-react';
import { useImageAnalysis } from '../hooks/useApi';

interface AnalysisResult {
  result: {
    isAuthentic: boolean;
    confidence: number;
    details: {
      isDeepfake: boolean;
      modelInfo: Record<string, any>;
    };
    metrics: {
      processingTime: number;
      modelVersion: string;
    };
  };
  error?: string;
  key_findings?: string[];
  detailed_analysis?: any;
  case_id?: string;
  analysis_date?: string;
  technical_details?: {
    processing_time_seconds?: number;
  };
}

const ImageAnalysis = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const { mutate: analyzeImage, isPending: isAnalyzing } = useImageAnalysis();

  // Handler functions
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange({ target: { files: e.dataTransfer.files } } as any);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid file type. Please upload an image (JPEG, PNG, WebP, GIF)');
      return;
    }

    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('File is too large. Maximum size is 50MB');
      return;
    }

    setSelectedFile(file);
    setError('');
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    
    setError('');
    setAnalysisResult(null);
    
    analyzeImage(selectedFile, {
      onSuccess: (result) => {
        if (!result?.result) {
          throw new Error('Invalid response from server');
        }

        const analysisResult = {
          result: {
            isAuthentic: result.result.isAuthentic,
            confidence: result.result.confidence,
            details: {
              isDeepfake: result.result.details.isDeepfake,
              modelInfo: result.result.details.modelInfo || {}
            },
            metrics: {
              processingTime: result.result.metrics?.processingTime || 0,
              modelVersion: result.result.metrics?.modelVersion || '1.0.0'
            }
          },
          key_findings: [
            result.result.isAuthentic 
              ? 'No signs of manipulation detected' 
              : 'Potential signs of manipulation found',
            `Confidence: ${(result.result.confidence * 100).toFixed(1)}%`,
            'Analysis completed successfully'
          ],
          case_id: result.id || `CASE_${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
          analysis_date: result.createdAt || new Date().toISOString(),
          technical_details: {
            model_version: result.result.metrics?.modelVersion || 'unknown',
            processing_time_seconds: result.result.metrics?.processingTime || 0,
            analysis_timestamp: result.createdAt || new Date().toISOString()
          }
        };

        setAnalysisResult(analysisResult);
      },
      onError: (err) => {
        console.error('Analysis failed:', err);
        setError(err.message || 'Failed to analyze image. Please try again.');
      }
    });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-3">
          <div className="w-14 h-14 bg-[#00bfff]/10 rounded-xl flex items-center justify-center border border-[#00bfff]/20">
            <img src="/image-icon.svg" alt="Image Analysis" className="w-7 h-7 text-[#00bfff]" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Image Analysis</h1>
            <p className="text-gray-400 text-sm">Advanced deepfake detection for images with 98.2% accuracy</p>
          </div>
        </div>

        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-[#00bfff]/10 rounded-xl flex items-center justify-center border border-[#00bfff]/20">
              <img src="/image-icon.svg" alt="Image Analysis" className="w-7 h-7 text-[#00bfff]" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Image Analysis</h1>
              <p className="text-gray-400 text-sm">Advanced deepfake detection for images with 98.2% accuracy</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full bg-green-500 animate-pulse`}></div>
            <span className="text-sm text-gray-400">Checking...</span>
          </div>
        </div>

        <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
          <div className="text-center mb-6">
            <h2 className="text-xl font-semibold text-white mb-2">Upload Image for Analysis</h2>
            <p className="text-gray-400 text-sm">Drag and drop your image here, or click to browse</p>
          </div>

          <div
            className={`border-2 border-dashed rounded-xl p-16 transition-all duration-200 ${
              dragActive
                ? 'border-[#00bfff] bg-[#00bfff]/5'
                : selectedFile
                  ? 'border-green-500/50 bg-green-500/5'
                  : error
                    ? 'border-red-500/50 bg-red-500/5'
                    : 'border-[#00bfff]/40 bg-[#1e2128] hover:bg-[#252930] hover:border-[#00bfff]/60'
            } ${selectedFile ? 'cursor-default' : 'cursor-pointer'}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !selectedFile && document.getElementById('imageFileInput')?.click()}
          >
            <input
              id="imageFileInput"
              type="file"
              accept="image/jpeg,image/jpg,image/png,image/webp,image/gif"
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
                    <p className="text-gray-400 text-sm">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAnalyze();
                      }}
                      disabled={isAnalyzing}
                      className="bg-[#00bfff] hover:bg-[#00a8e6] disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Eye className="w-5 h-5" />
                          Analyze Image
                        </>
                      )}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedFile(null);
                        setAnalysisResult(null);
                        setError('');
                      }}
                      disabled={isAnalyzing}
                      className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                    >
                      <Upload className="w-5 h-5" />
                      Change File
                    </button>
                  </div>
                </>
              ) : error ? (
                <>
                  <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center">
                    <AlertCircle className="w-10 h-10 text-red-500" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-red-400 mb-1">Invalid File</p>
                    <p className="text-gray-400 text-sm">{error}</p>
                  </div>
                  <button className="bg-[#00bfff] hover:bg-[#00a8e6] text-white px-8 py-3 rounded-lg font-medium transition-all duration-200">
                    Choose Another File
                  </button>
                </>
              ) : (
                <>
                  <div className="w-20 h-20 bg-[#00bfff]/10 rounded-full flex items-center justify-center">
                    <Camera className="w-10 h-10 text-[#00bfff]" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white mb-1">Drop your image here</p>
                    <p className="text-gray-400 text-sm">Supports JPG, PNG, WebP, GIF up to 50MB</p>
                  </div>
                  <button className="bg-[#00bfff] hover:bg-[#00a8e6] text-white px-8 py-3 rounded-lg font-medium transition-all duration-200">
                    Choose File
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4 text-sm text-gray-400">
            <svg
              className="w-4 h-4 text-[#00bfff]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>Only image files are accepted • Maximum file size: 50MB</span>
          </div>
        </div>

        {/* Analysis Results */}
        {analysisResult?.result && !isAnalyzing && (
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
            <div className="flex items-center gap-4 mb-6">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  analysisResult.result.isAuthentic
                    ? 'bg-green-500/10 border-2 border-green-500/30'
                    : 'bg-red-500/10 border-2 border-red-500/30'
                }`}
              >
                {analysisResult.result.isAuthentic ? (
                  <CheckCircle className="w-8 h-8 text-green-500" />
                ) : (
                  <AlertCircle className="w-8 h-8 text-red-500" />
                )}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">
                  {analysisResult.result.isAuthentic ? 'Authentic Media' : 'Potential Deepfake'}
                </h2>
                <p className="text-gray-400">
                  Confidence:{' '}
                  <span className="text-white font-semibold">
                    {(analysisResult.result.confidence * 100).toFixed(1)}%
                  </span>
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <FileText className="w-5 h-5 text-[#00bfff]" />
                  Key Findings
                </h3>
                <div className="space-y-2">
                  {analysisResult.key_findings?.map((finding, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-[#00bfff] rounded-full mt-2 flex-shrink-0" />
                      <p className="text-gray-300 text-sm">{finding}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#00bfff]" />
                  Analysis Scores
                </h3>
                <div className="space-y-3">
                  {analysisResult.technical_details?.processing_time_seconds && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">Processing Time</span>
                      <span className="text-white font-semibold">
                        {analysisResult.result.metrics.processingTime.toFixed(2)}s
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-[#00bfff]/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Quick Analysis</h3>
            <p className="text-gray-400 text-sm mb-4">Fast detection using core models</p>
            <div className="text-3xl font-bold text-[#00bfff]">~30s</div>
          </div>

          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-[#00bfff]/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Comprehensive</h3>
            <p className="text-gray-400 text-sm mb-4">Deep analysis with all models</p>
            <div className="text-3xl font-bold text-[#00bfff]">~2min</div>
          </div>

          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-6 hover:border-[#00bfff]/50 transition-colors">
            <h3 className="text-lg font-semibold text-white mb-2">Forensic Grade</h3>
            <p className="text-lg font-semibold text-white mb-2">Maximum accuracy analysis</p>
            <div className="text-3xl font-bold text-[#00bfff]">~5min</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageAnalysis;