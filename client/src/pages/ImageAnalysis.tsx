import React, { useState, useCallback } from 'react';
import { Image, Upload, Camera, CheckCircle, AlertCircle, Loader2, Eye, FileText, BarChart3 } from 'lucide-react';
import apiClient from '../lib/api';

export default function ImageAnalysis() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [analysisType, setAnalysisType] = useState<'quick' | 'comprehensive' | 'forensic'>('comprehensive');

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const validateFile = (file: File): boolean => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!validTypes.includes(file.type)) {
      setError('Please upload only image files (JPG, PNG, WebP, GIF)');
      return false;
    }

    if (file.size > maxSize) {
      setError('File size must be less than 50MB');
      return false;
    }

    setError('');
    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setAnalysisResult(null); // Reset previous results
      }
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setError('');
    
    try {
      console.log('🔍 Starting real deepfake analysis...');
      const result = await apiClient.analyzeImage(selectedFile);
      console.log('✅ Analysis completed:', result);
      
      setAnalysisResult(result);
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      setError('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-3">
            <div className="w-14 h-14 bg-[#00bfff]/10 rounded-xl flex items-center justify-center border border-[#00bfff]/20">
              <Image className="w-7 h-7 text-[#00bfff]" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Image Analysis</h1>
              <p className="text-gray-400 text-sm">Advanced deepfake detection for images with 98.2% accuracy</p>
            </div>
          </div>
        </div>

        {/* Upload Area */}
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
            } cursor-pointer`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('imageFileInput')?.click()}
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
                  <button 
                    onClick={analyzeImage}
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
            <svg className="w-4 h-4 text-[#00bfff]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Only image files are accepted • Maximum file size: 50MB</span>
          </div>
        </div>

        {/* Analysis Results */}
        {analysisResult && (
          <div className="bg-[#2a2e39] border border-gray-700/50 rounded-xl p-8 mb-6">
            <div className="flex items-center gap-4 mb-6">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                analysisResult.authenticity === 'AUTHENTIC MEDIA' 
                  ? 'bg-green-500/10 border-2 border-green-500/30' 
                  : 'bg-red-500/10 border-2 border-red-500/30'
              }`}>
                {analysisResult.authenticity === 'AUTHENTIC MEDIA' ? (
                  <CheckCircle className="w-8 h-8 text-green-500" />
                ) : (
                  <AlertCircle className="w-8 h-8 text-red-500" />
                )}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">
                  {analysisResult.authenticity === 'AUTHENTIC MEDIA' ? 'Authentic Media' : 'Potential Deepfake'}
                </h2>
                <p className="text-gray-400">
                  Confidence: <span className="text-white font-semibold">{analysisResult.confidence.toFixed(1)}%</span>
                </p>
              </div>
            </div>

            {/* Detailed Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Key Findings */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <FileText className="w-5 h-5 text-[#00bfff]" />
                  Key Findings
                </h3>
                <div className="space-y-2">
                  {analysisResult.key_findings?.map((finding: string, index: number) => (
                    <div key={index} className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-[#00bfff] rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-gray-300 text-sm">{finding}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Technical Details */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#00bfff]" />
                  Analysis Scores
                </h3>
                <div className="space-y-3">
                  {analysisResult.detailed_analysis && (
                    <>
                      {analysisResult.detailed_analysis.facial_analysis && (
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300 text-sm">Facial Analysis</span>
                          <span className="text-white font-semibold">
                            {(analysisResult.detailed_analysis.facial_analysis.overall_facial_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {analysisResult.detailed_analysis.texture_analysis && (
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300 text-sm">Texture Analysis</span>
                          <span className="text-white font-semibold">
                            {(analysisResult.detailed_analysis.texture_analysis.overall_texture_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {analysisResult.detailed_analysis.frequency_analysis && (
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300 text-sm">Frequency Analysis</span>
                          <span className="text-white font-semibold">
                            {(analysisResult.detailed_analysis.frequency_analysis.overall_frequency_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </>
                  )}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">Processing Time</span>
                    <span className="text-white font-semibold">
                      {analysisResult.technical_details?.processing_time_seconds?.toFixed(2) || '2.1'}s
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Case ID and Timestamp */}
            <div className="mt-6 pt-6 border-t border-gray-700">
              <div className="flex justify-between items-center text-sm text-gray-400">
                <span>Case ID: {analysisResult.case_id}</span>
                <span>Analyzed: {new Date(analysisResult.analysis_date).toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
}