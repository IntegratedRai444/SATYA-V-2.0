<<<<<<< HEAD
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  Upload, 
  Image, 
  Video, 
  Music, 
  FileText, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Camera,
  FolderOpen,
  Play,
  Square,
  Download,
  Search,
  Filter,
  Trash2,
  Eye,
  Share2
} from 'lucide-react';
import { useToast } from '../../hooks/use-toast';
import { createApiUrl } from '../../lib/config';

interface AnalysisResult {
  authenticity: string;
  confidence: number;
  analysis_date: string;
  case_id: string;
  neural_network_scores?: {
    resnet50?: number;
    efficientnet?: number;
    ensemble?: number;
    vision_transformer?: number;
  };
  face_analysis?: {
    faces_detected: number;
    encoding_quality: number;
    face_consistency: number;
    facial_landmarks?: number;
    expression_analysis?: string;
  };
  texture_analysis?: {
    compression_artifacts: number;
    noise_level: number;
    edge_consistency: number;
    color_consistency: number;
  };
  metadata_analysis?: {
    exif_data: string;
    camera_model: string;
    timestamp: string;
    gps_data: string;
  };
  key_findings: string[];
  technical_details: {
    models_used: string[];
    device: string;
    analysis_version: string;
    neural_architectures?: string[];
    feature_dimensions?: string;
    computation_type?: string;
  };
  risk_assessment?: {
    overall_risk: string;
    manipulation_probability: number;
    confidence_level: string;
    recommendations: string[];
  };
  processing_time_ms?: number;
  analysis_type?: string;
}

interface VideoAnalysisResult extends AnalysisResult {
  video_analysis?: {
    total_frames_analyzed: number;
    video_duration_seconds: number;
    frame_rate: number;
    manipulation_consistency: number;
    temporal_analysis: string;
  };
  frame_analyses?: AnalysisResult[];
}

interface AudioAnalysisResult extends AnalysisResult {
  audio_analysis?: {
    synthesis_detection: number;
    spectrogram_quality: number;
    frequency_analysis: string;
    audio_duration_seconds: number;
    sample_rate: number;
    frequency_range: string;
    spectral_features: string;
  };
}

interface BatchAnalysisResult {
  batch_id: string;
  total_files: number;
  analysis_date: string;
  average_confidence: number;
  results: AnalysisResult[];
  batch_summary: {
    authentic_files: number;
    manipulated_files: number;
    processing_time_total: number;
  };
}

export default function UploadSection() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<(AnalysisResult | VideoAnalysisResult | AudioAnalysisResult | BatchAnalysisResult)[]>([]);
  const [currentAnalysis, setCurrentAnalysis] = useState<string>('');
  const [analysisHistory, setAnalysisHistory] = useState<any[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterResult, setFilterResult] = useState('all');
  
  // Webcam states
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  
  // Batch analysis states
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchProgress, setBatchProgress] = useState<{[key: string]: number}>({});
  const [batchResults, setBatchResults] = useState<any[]>([]);
  
  // Advanced analysis options
  const [analysisType, setAnalysisType] = useState('comprehensive');
  const [confidenceThreshold, setConfidenceThreshold] = useState(80);
  const [enableAdvancedModels, setEnableAdvancedModels] = useState(true);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();

  // Add state for selected tab
  const [mediaTab, setMediaTab] = useState<'image' | 'video' | 'audio' | 'webcam'>('image');

  // Add video preview state
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  // Add audio preview state
  const [audioPreview, setAudioPreview] = useState<string | null>(null);

  // Webcam functionality
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      
      setWebcamStream(stream);
      setIsWebcamActive(true);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      toast({
        title: "Webcam activated",
        description: "Position your face in the frame for analysis"
      });
    } catch (error) {
      console.error('Error accessing webcam:', error);
      toast({
        title: "Webcam Error",
        description: "Failed to access webcam. Please check permissions.",
        variant: "destructive"
      });
    }
  }, [toast]);

  const stopWebcam = useCallback(() => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
    setIsWebcamActive(false);
    setCapturedImage(null);
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [webcamStream]);

  const captureWebcamImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) return null;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImage(imageData);
    
    return imageData;
  }, []);

  const analyzeWebcamImage = useCallback(async () => {
    if (!capturedImage) {
      toast({
        title: "No Image Captured",
        description: "Please capture an image first",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setCurrentAnalysis('Analyzing webcam image...');

    try {
      const apiUrl = await createApiUrl('/api/ai/analyze/image');
      const formData = new FormData();
      formData.append('imageData', capturedImage);
      formData.append('analysis_type', analysisType);
      formData.append('confidence_threshold', confidenceThreshold.toString());
      formData.append('enable_advanced_models', enableAdvancedModels.toString());

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setResults([result]);
      setProgress(100);

      toast({
        title: `${result.authenticity}`,
        description: `Confidence: ${result.confidence}% - ${result.key_findings[0]}`,
      });
    } catch (error) {
      console.error('Webcam analysis failed:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze webcam image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      setCurrentAnalysis('');
    }
  }, [capturedImage, analysisType, confidenceThreshold, enableAdvancedModels, toast]);

  // Load analysis history
  const loadAnalysisHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const apiUrl = await createApiUrl('/api/scans');
      const response = await fetch(apiUrl);
      
      if (response.ok) {
        const history = await response.json();
        setAnalysisHistory(history);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
      toast({
        title: "History Load Failed",
        description: "Failed to load analysis history",
        variant: "destructive"
      });
    } finally {
      setIsLoadingHistory(false);
    }
  }, [toast]);

  // Filter history results
  const filteredHistory = analysisHistory.filter(item => {
    const matchesSearch = searchQuery === '' || 
      item.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === 'all' || item.type === filterType;
    const matchesResult = filterResult === 'all' || item.result === filterResult;
    
    return matchesSearch && matchesType && matchesResult;
  });

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [webcamStream]);

  const analyzeFile = async (file: File, analysisType: string = 'comprehensive') => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('analysis_type', analysisType);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('enable_advanced_models', enableAdvancedModels.toString());

    try {
      const apiUrl = await createApiUrl('/api/ai/analyze/image');
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  };

  const analyzeVideo = async (file: File) => {
    const formData = new FormData();
    formData.append('video', file);
    formData.append('analysis_type', analysisType);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('enable_advanced_models', enableAdvancedModels.toString());

    try {
      const apiUrl = await createApiUrl('/api/ai/analyze/video');
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Video analysis failed:', error);
      throw error;
    }
  };

  const analyzeAudio = async (file: File) => {
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('analysis_type', analysisType);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('enable_advanced_models', enableAdvancedModels.toString());

    try {
      const apiUrl = await createApiUrl('/api/ai/analyze/audio');
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Audio analysis failed:', error);
      throw error;
    }
  };

  const batchAnalyze = async (files: File[]) => {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append('files', file);
    });
    formData.append('analysis_type', analysisType);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('enable_advanced_models', enableAdvancedModels.toString());

    try {
      const apiUrl = await createApiUrl('/api/ai/analyze/batch');
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Batch analysis failed:', error);
      throw error;
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    const maxFileSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxFileSize) {
      toast({
        title: "File Size Error",
        description: `File exceeds the 100MB limit`,
        variant: "destructive"
      });
      return;
    }
    setIsAnalyzing(true);
    setProgress(0);
    setResults([]);
    setVideoPreview(null);
    setAudioPreview(null);
    try {
      let result;
      if (mediaTab === 'image') {
        if (!file.type.startsWith('image/')) throw new Error('Please upload an image file.');
        result = await analyzeFile(file, analysisType);
      } else if (mediaTab === 'video') {
        if (!file.type.startsWith('video/')) throw new Error('Please upload a video file.');
        setVideoPreview(URL.createObjectURL(file));
        result = await analyzeVideo(file);
      } else if (mediaTab === 'audio') {
        if (!file.type.startsWith('audio/')) throw new Error('Please upload an audio file.');
        setAudioPreview(URL.createObjectURL(file));
        result = await analyzeAudio(file);
      }
      setProgress(100);
      setResults([result]);
      toast({
        title: `${result.authenticity}`,
        description: `Confidence: ${result.confidence}% - ${result.key_findings?.[0]}`,
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : 'Failed to analyze the file. Please try again.',
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      setCurrentAnalysis('');
    }
  }, [mediaTab, toast, analysisType]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp'],
      'video/*': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'],
      'audio/*': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    },
    multiple: true,
    disabled: isAnalyzing
  });

  const renderAnalysisResult = (result: AnalysisResult | VideoAnalysisResult | AudioAnalysisResult | BatchAnalysisResult, index: number) => {
    if ('batch_id' in result) {
      // Batch analysis result
      const batchResult = result as BatchAnalysisResult;
      return (
        <Card key={batchResult.batch_id} className="mb-4">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              Batch Analysis Results
            </CardTitle>
            <CardDescription>
              Analyzed {batchResult.total_files} files on {batchResult.analysis_date}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{batchResult.batch_summary.authentic_files}</div>
                <div className="text-sm text-muted-foreground">Authentic Files</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">{batchResult.batch_summary.manipulated_files}</div>
                <div className="text-sm text-muted-foreground">Manipulated Files</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{batchResult.average_confidence}%</div>
                <div className="text-sm text-muted-foreground">Avg Confidence</div>
              </div>
            </div>
            
            <div className="space-y-2">
              {batchResult.results.map((fileResult, fileIndex) => (
                <div key={fileIndex} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">File {fileIndex + 1}</span>
                    <Badge variant={fileResult.authenticity === 'AUTHENTIC MEDIA' ? 'default' : 'destructive'}>
                      {fileResult.authenticity}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Confidence: {fileResult.confidence}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      );
    }

    // Single file analysis result
    const singleResult = result as AnalysisResult | VideoAnalysisResult | AudioAnalysisResult;
    
    return (
      <Card key={singleResult.case_id} className="mb-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {singleResult.analysis_type?.includes('video') ? (
              <Video className="h-5 w-5" />
            ) : singleResult.analysis_type?.includes('audio') ? (
              <Music className="h-5 w-5" />
            ) : (
              <Image className="h-5 w-5" />
            )}
            Analysis Result
          </CardTitle>
          <CardDescription>
            Case ID: {singleResult.case_id} • {singleResult.analysis_date}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Main Result */}
            <div>
              <h3 className="font-semibold mb-2">Main Result</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span>Authenticity:</span>
                  <Badge variant={singleResult.authenticity === 'AUTHENTIC MEDIA' ? 'default' : 'destructive'}>
                    {singleResult.authenticity}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span>Confidence:</span>
                  <span className="font-medium">{singleResult.confidence}%</span>
                </div>
                {singleResult.processing_time_ms && (
                  <div className="flex items-center justify-between">
                    <span>Processing Time:</span>
                    <span className="font-medium">{singleResult.processing_time_ms}ms</span>
                  </div>
                )}
              </div>
            </div>

            {/* Neural Network Scores */}
            {singleResult.neural_network_scores && (
              <div>
                <h3 className="font-semibold mb-2">Neural Network Scores</h3>
                <div className="space-y-2">
                  {Object.entries(singleResult.neural_network_scores).map(([model, score]) => (
                    <div key={model} className="flex items-center justify-between">
                      <span className="capitalize">{model.replace('_', ' ')}:</span>
                      <span className="font-medium">{score}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Face Analysis */}
          {singleResult.face_analysis && (
            <div className="mt-6">
              <h3 className="font-semibold mb-2">Face Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Faces Detected</div>
                  <div className="font-medium">{singleResult.face_analysis.faces_detected}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Encoding Quality</div>
                  <div className="font-medium">{singleResult.face_analysis.encoding_quality}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Face Consistency</div>
                  <div className="font-medium">{singleResult.face_analysis.face_consistency}%</div>
                </div>
                {singleResult.face_analysis.facial_landmarks && (
                  <div>
                    <div className="text-sm text-muted-foreground">Facial Landmarks</div>
                    <div className="font-medium">{singleResult.face_analysis.facial_landmarks}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Texture Analysis */}
          {singleResult.texture_analysis && (
            <div className="mt-6">
              <h3 className="font-semibold mb-2">Texture Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Compression Artifacts</div>
                  <div className="font-medium">{singleResult.texture_analysis.compression_artifacts}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Noise Level</div>
                  <div className="font-medium">{singleResult.texture_analysis.noise_level}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Edge Consistency</div>
                  <div className="font-medium">{singleResult.texture_analysis.edge_consistency}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Color Consistency</div>
                  <div className="font-medium">{singleResult.texture_analysis.color_consistency}%</div>
                </div>
              </div>
            </div>
          )}

          {/* Video Analysis */}
          {'video_analysis' in singleResult && singleResult.video_analysis && (
            <div className="mt-6">
              <h3 className="font-semibold mb-2">Video Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Frames Analyzed</div>
                  <div className="font-medium">{singleResult.video_analysis.total_frames_analyzed}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Duration</div>
                  <div className="font-medium">{singleResult.video_analysis.video_duration_seconds.toFixed(1)}s</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Frame Rate</div>
                  <div className="font-medium">{singleResult.video_analysis.frame_rate} fps</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Manipulation Consistency</div>
                  <div className="font-medium">{singleResult.video_analysis.manipulation_consistency}%</div>
                </div>
              </div>
            </div>
          )}

          {/* Audio Analysis */}
          {'audio_analysis' in singleResult && singleResult.audio_analysis && (
            <div className="mt-6">
              <h3 className="font-semibold mb-2">Audio Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Synthesis Detection</div>
                  <div className="font-medium">{singleResult.audio_analysis.synthesis_detection}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Spectrogram Quality</div>
                  <div className="font-medium">{singleResult.audio_analysis.spectrogram_quality}%</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Duration</div>
                  <div className="font-medium">{singleResult.audio_analysis.audio_duration_seconds.toFixed(1)}s</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Sample Rate</div>
                  <div className="font-medium">{singleResult.audio_analysis.sample_rate}Hz</div>
                </div>
              </div>
            </div>
          )}

          {/* Risk Assessment */}
          {singleResult.risk_assessment && (
            <div className="mt-6">
              <h3 className="font-semibold mb-2">Risk Assessment</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span>Overall Risk:</span>
                  <Badge variant={
                    singleResult.risk_assessment.overall_risk === 'LOW' ? 'default' :
                    singleResult.risk_assessment.overall_risk === 'MEDIUM' ? 'secondary' : 'destructive'
                  }>
                    {singleResult.risk_assessment.overall_risk}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span>Manipulation Probability:</span>
                  <span className="font-medium">{singleResult.risk_assessment.manipulation_probability}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Confidence Level:</span>
                  <span className="font-medium">{singleResult.risk_assessment.confidence_level}</span>
                </div>
              </div>
              
              {singleResult.risk_assessment.recommendations && (
                <div className="mt-3">
                  <div className="text-sm font-medium mb-1">Recommendations:</div>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    {singleResult.risk_assessment.recommendations.map((rec, idx) => (
                      <li key={idx}>• {rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Key Findings */}
          <div className="mt-6">
            <h3 className="font-semibold mb-2">Key Findings</h3>
            <div className="space-y-1">
              {singleResult.key_findings.map((finding, idx) => (
                <div key={idx} className="flex items-start gap-2 text-sm">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>{finding}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Technical Details */}
          <div className="mt-6">
            <h3 className="font-semibold mb-2">Technical Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Models Used:</span>
                <div className="font-medium">
                  {(singleResult.technical_details?.models_used && Array.isArray(singleResult.technical_details.models_used))
                    ? singleResult.technical_details.models_used.join(', ')
                    : 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-muted-foreground">Device:</span>
                <div className="font-medium">
                  {singleResult.technical_details?.device ?? 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-muted-foreground">Version:</span>
                <div className="font-medium">{singleResult.technical_details?.analysis_version ?? 'N/A'}</div>
              </div>
              {singleResult.technical_details?.neural_architectures && Array.isArray(singleResult.technical_details.neural_architectures) ? (
                <div>
                  <span className="text-muted-foreground">Architectures:</span>
                  <div className="font-medium">{singleResult.technical_details.neural_architectures.join(', ')}</div>
                </div>
              ) : null}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      <Tabs defaultValue={mediaTab} value={mediaTab} onValueChange={setMediaTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="image">Image</TabsTrigger>
          <TabsTrigger value="video">Video</TabsTrigger>
          <TabsTrigger value="audio">Audio</TabsTrigger>
          <TabsTrigger value="webcam">Webcam</TabsTrigger>
        </TabsList>

        {/* Image Tab */}
        <TabsContent value="image" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Image for Analysis
              </CardTitle>
              <CardDescription>
                Drag and drop or click to upload images for AI deepfake detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Advanced Analysis Options */}
              <div className="space-y-4 p-4 border rounded-lg bg-muted/25">
                <h3 className="text-sm font-medium">Advanced Analysis Options</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="image-analysis-type" className="text-sm text-muted-foreground">Analysis Type</label>
                    <select
                      id="image-analysis-type"
                      value={analysisType}
                      onChange={(e) => setAnalysisType(e.target.value)}
                      className="w-full mt-1 p-2 border rounded-md"
                      aria-label="Select analysis type"
                    >
                      <option value="comprehensive">Comprehensive (Recommended)</option>
                      <option value="quick">Quick Scan (Fast)</option>
                      <option value="detailed">Detailed Analysis (Thorough)</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="image-confidence-threshold" className="text-sm text-muted-foreground">Confidence Threshold</label>
                    <input
                      id="image-confidence-threshold"
                      type="range"
                      min="50"
                      max="95"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                      className="w-full mt-1"
                      aria-label="Set confidence threshold"
                    />
                    <span className="text-xs text-muted-foreground">{confidenceThreshold}%</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="image-advanced-models"
                      checked={enableAdvancedModels}
                      onChange={(e) => setEnableAdvancedModels(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="image-advanced-models" className="text-sm">
                      Enable Advanced Models
                    </label>
                  </div>
                </div>
              </div>

              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-primary/50'
                } ${isAnalyzing ? 'pointer-events-none opacity-50' : ''}`}
              >
                <input {...getInputProps()} accept="image/*" />
                <div className="flex flex-col items-center gap-4">
                  {isAnalyzing ? (
                    <Loader2 className="h-12 w-12 animate-spin text-primary" />
                  ) : (
                    <Upload className="h-12 w-12 text-muted-foreground" />
                  )}
                  <div>
                    <p className="text-lg font-medium">
                      {isAnalyzing ? currentAnalysis : 'Drop image here or click to upload'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Supports images (JPEG, PNG, GIF, BMP, WebP)
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Maximum file size: 100MB
                    </p>
                  </div>
                </div>
              </div>

              {isAnalyzing && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Analyzing...</span>
                    <span className="text-sm text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Video Tab */}
        <TabsContent value="video" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Video className="h-5 w-5" />
                Upload Video for Analysis
              </CardTitle>
              <CardDescription>
                Drag and drop or click to upload video files for AI deepfake detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Advanced Analysis Options */}
              <div className="space-y-4 p-4 border rounded-lg bg-muted/25">
                <h3 className="text-sm font-medium">Advanced Analysis Options</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="video-analysis-type" className="text-sm text-muted-foreground">Analysis Type</label>
                    <select
                      id="video-analysis-type"
                      value={analysisType}
                      onChange={(e) => setAnalysisType(e.target.value)}
                      className="w-full mt-1 p-2 border rounded-md"
                      aria-label="Select analysis type"
                    >
                      <option value="comprehensive">Comprehensive (Recommended)</option>
                      <option value="quick">Quick Scan (Fast)</option>
                      <option value="detailed">Detailed Analysis (Thorough)</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="video-confidence-threshold" className="text-sm text-muted-foreground">Confidence Threshold</label>
                    <input
                      id="video-confidence-threshold"
                      type="range"
                      min="50"
                      max="95"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                      className="w-full mt-1"
                      aria-label="Set confidence threshold"
                    />
                    <span className="text-xs text-muted-foreground">{confidenceThreshold}%</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="video-advanced-models"
                      checked={enableAdvancedModels}
                      onChange={(e) => setEnableAdvancedModels(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="video-advanced-models" className="text-sm">
                      Enable Advanced Models
                    </label>
                  </div>
                </div>
              </div>

              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-primary/50'
                } ${isAnalyzing ? 'pointer-events-none opacity-50' : ''}`}
              >
                <input {...getInputProps()} accept="video/*" />
                <div className="flex flex-col items-center gap-4">
                  {isAnalyzing ? (
                    <Loader2 className="h-12 w-12 animate-spin text-primary" />
                  ) : (
                    <Video className="h-12 w-12 text-muted-foreground" />
                  )}
                  <div>
                    <p className="text-lg font-medium">
                      {isAnalyzing ? currentAnalysis : 'Drop video here or click to upload'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Supports videos (MP4, AVI, MOV, WMV, FLV, WebM)
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Maximum file size: 100MB
                    </p>
                  </div>
                </div>
              </div>

              {videoPreview && (
                <video src={videoPreview} controls className="w-full rounded-lg mt-4" />
              )}

              {isAnalyzing && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Analyzing...</span>
                    <span className="text-sm text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Audio Tab */}
        <TabsContent value="audio" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Music className="h-5 w-5" />
                Upload Audio for Analysis
              </CardTitle>
              <CardDescription>
                Drag and drop or click to upload audio files for AI deepfake detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Advanced Analysis Options */}
              <div className="space-y-4 p-4 border rounded-lg bg-muted/25">
                <h3 className="text-sm font-medium">Advanced Analysis Options</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="audio-analysis-type" className="text-sm text-muted-foreground">Analysis Type</label>
                    <select
                      id="audio-analysis-type"
                      value={analysisType}
                      onChange={(e) => setAnalysisType(e.target.value)}
                      className="w-full mt-1 p-2 border rounded-md"
                      aria-label="Select analysis type"
                    >
                      <option value="comprehensive">Comprehensive (Recommended)</option>
                      <option value="quick">Quick Scan (Fast)</option>
                      <option value="detailed">Detailed Analysis (Thorough)</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="audio-confidence-threshold" className="text-sm text-muted-foreground">Confidence Threshold</label>
                    <input
                      id="audio-confidence-threshold"
                      type="range"
                      min="50"
                      max="95"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                      className="w-full mt-1"
                      aria-label="Set confidence threshold"
                    />
                    <span className="text-xs text-muted-foreground">{confidenceThreshold}%</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="audio-advanced-models"
                      checked={enableAdvancedModels}
                      onChange={(e) => setEnableAdvancedModels(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="audio-advanced-models" className="text-sm">
                      Enable Advanced Models
                    </label>
                  </div>
                </div>
              </div>

              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-primary/50'
                } ${isAnalyzing ? 'pointer-events-none opacity-50' : ''}`}
              >
                <input {...getInputProps()} accept="audio/*" />
                <div className="flex flex-col items-center gap-4">
                  {isAnalyzing ? (
                    <Loader2 className="h-12 w-12 animate-spin text-primary" />
                  ) : (
                    <Music className="h-12 w-12 text-muted-foreground" />
                  )}
                  <div>
                    <p className="text-lg font-medium">
                      {isAnalyzing ? currentAnalysis : 'Drop audio here or click to upload'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Supports audio (MP3, WAV, FLAC, AAC, OGG, M4A)
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Maximum file size: 100MB
                    </p>
                  </div>
                </div>
              </div>

              {audioPreview && (
                <audio src={audioPreview} controls className="w-full rounded-lg mt-4" />
              )}

              {isAnalyzing && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Analyzing...</span>
                    <span className="text-sm text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Webcam Tab (existing logic) */}
        <TabsContent value="webcam" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Webcam Analysis
              </CardTitle>
              <CardDescription>
                Capture and analyze images directly from your webcam using advanced AI detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Webcam Controls */}
              <div className="flex gap-2">
                <Button 
                  onClick={isWebcamActive ? stopWebcam : startWebcam}
                  variant={isWebcamActive ? "destructive" : "default"}
                  disabled={isAnalyzing}
                >
                  {isWebcamActive ? (
                    <>
                      <Square className="mr-2 h-4 w-4" />
                      Stop Webcam
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Webcam
                    </>
                  )}
                </Button>
                
                {isWebcamActive && (
                  <Button 
                    onClick={captureWebcamImage}
                    variant="outline"
                    disabled={isAnalyzing}
                  >
                <Camera className="mr-2 h-4 w-4" />
                    Capture Image
              </Button>
                )}
                
                {capturedImage && (
                  <Button 
                    onClick={analyzeWebcamImage}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Eye className="mr-2 h-4 w-4" />
                    )}
                    Analyze Image
                  </Button>
                )}
              </div>

              {/* Webcam Video Feed */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Live Feed</h3>
                  <div className="relative aspect-video bg-muted rounded-md overflow-hidden">
                    {isWebcamActive ? (
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                          <Camera className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            Click "Start Webcam" to begin
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Captured Image Preview */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Captured Image</h3>
                  <div className="relative aspect-video bg-muted rounded-md overflow-hidden">
                    {capturedImage ? (
                      <img
                        src={capturedImage}
                        alt="Captured"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                          <Image className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            Capture an image to analyze
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Hidden canvas for capture */}
              <canvas ref={canvasRef} className="hidden" />

              {/* Analysis Progress */}
              {isAnalyzing && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{currentAnalysis}</span>
                    <span className="text-sm text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}

              {/* Analysis Options */}
              <div className="space-y-4 pt-4 border-t">
                <h3 className="text-sm font-medium">Analysis Options</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="analysis-type" className="text-sm text-muted-foreground">Analysis Type</label>
                    <select
                      id="analysis-type"
                      value={analysisType}
                      onChange={(e) => setAnalysisType(e.target.value)}
                      className="w-full mt-1 p-2 border rounded-md"
                      aria-label="Select analysis type"
                    >
                      <option value="comprehensive">Comprehensive</option>
                      <option value="quick">Quick Scan</option>
                      <option value="detailed">Detailed Analysis</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="confidence-threshold" className="text-sm text-muted-foreground">Confidence Threshold</label>
                    <input
                      id="confidence-threshold"
                      type="range"
                      min="50"
                      max="95"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                      className="w-full mt-1"
                      aria-label="Set confidence threshold"
                    />
                    <span className="text-xs text-muted-foreground">{confidenceThreshold}%</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="advanced-models"
                      checked={enableAdvancedModels}
                      onChange={(e) => setEnableAdvancedModels(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="advanced-models" className="text-sm">
                      Enable Advanced Models
                    </label>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Analysis History
              </CardTitle>
              <CardDescription>
                View your recent analysis results and reports
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* History Controls */}
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <input
                      type="text"
                      placeholder="Search by filename..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 border rounded-md"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="px-3 py-2 border rounded-md"
                    aria-label="Filter by media type"
                  >
                    <option value="all">All Types</option>
                    <option value="image">Images</option>
                    <option value="video">Videos</option>
                    <option value="audio">Audio</option>
                  </select>
                  <select
                    value={filterResult}
                    onChange={(e) => setFilterResult(e.target.value)}
                    className="px-3 py-2 border rounded-md"
                    aria-label="Filter by result"
                  >
                    <option value="all">All Results</option>
                    <option value="authentic">Authentic</option>
                    <option value="deepfake">Deepfake</option>
                  </select>
                  <Button onClick={loadAnalysisHistory} variant="outline">
                    <Filter className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* History Table */}
              <div className="border rounded-md">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left p-3 font-medium">Filename</th>
                        <th className="text-left p-3 font-medium">Type</th>
                        <th className="text-left p-3 font-medium">Result</th>
                        <th className="text-left p-3 font-medium">Confidence</th>
                        <th className="text-left p-3 font-medium">Date</th>
                        <th className="text-left p-3 font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {isLoadingHistory ? (
                        <tr>
                          <td colSpan={6} className="p-3 text-center">
                            <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                            <p className="text-sm text-muted-foreground mt-2">Loading history...</p>
                          </td>
                        </tr>
                      ) : filteredHistory.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="p-3 text-center text-muted-foreground">
                            No analysis history found
                          </td>
                        </tr>
                      ) : (
                        filteredHistory.map((item, index) => (
                          <tr key={index} className="border-t hover:bg-muted/25">
                            <td className="p-3">
                              <div className="flex items-center gap-2">
                                {item.type === 'image' && <Image className="h-4 w-4" />}
                                {item.type === 'video' && <Video className="h-4 w-4" />}
                                {item.type === 'audio' && <Music className="h-4 w-4" />}
                                <span className="font-medium">{item.filename}</span>
                              </div>
                            </td>
                            <td className="p-3">
                              <Badge variant="outline" className="capitalize">
                                {item.type}
                              </Badge>
                            </td>
                            <td className="p-3">
                              <Badge 
                                variant={item.result === 'authentic' ? 'default' : 'destructive'}
                              >
                                {item.result === 'authentic' ? 'Authentic' : 'Deepfake'}
                              </Badge>
                            </td>
                            <td className="p-3">
                              <span className="font-medium">{item.confidenceScore}%</span>
                            </td>
                            <td className="p-3 text-sm text-muted-foreground">
                              {new Date(item.createdAt).toLocaleDateString()}
                            </td>
                            <td className="p-3">
                              <div className="flex gap-1">
                                <Button size="sm" variant="ghost">
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button size="sm" variant="ghost">
                                  <Download className="h-4 w-4" />
                                </Button>
                                <Button size="sm" variant="ghost">
                                  <Share2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Load History Button */}
              {!isLoadingHistory && analysisHistory.length === 0 && (
                <div className="text-center">
                  <Button onClick={loadAnalysisHistory}>
                    <FileText className="mr-2 h-4 w-4" />
                    Load Analysis History
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Results Section */}
      {results.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Analysis Results</h2>
          {results.map((result, index) => renderAnalysisResult(result, index))}
=======
import { useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { Upload, Image, Video, Mic, Camera, Info, ScanLine } from "lucide-react";
import { useNavigation } from "@/hooks/useNavigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { apiRequest } from "@/lib/queryClient";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";

export default function UploadSection() {
  const [location, setLocation] = useLocation();
  const { navigate } = useNavigation();
  const { toast } = useToast();
  
  // Get the active tab from URL query parameter
  const searchParams = new URLSearchParams(location.split("?")[1] || "");
  const defaultType = searchParams.get("type") || "image";
  
  const [activeTab, setActiveTab] = useState(defaultType);
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update URL when tab changes
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setLocation(`/scan?type=${value}`, { replace: true });
  };

  // Handle file upload via browse button
  const handleBrowseFiles = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const selectedFiles = Array.from(e.target.files);
    
    try {
      // Validate file sizes
      const maxSize = activeTab === 'video' ? 50 * 1024 * 1024 : 10 * 1024 * 1024; // 50MB for video, 10MB for others
      
      const invalidFiles = selectedFiles.filter(file => file.size > maxSize);
      if (invalidFiles.length > 0) {
        toast({
          title: "File too large",
          description: `Maximum file size is ${maxSize / (1024 * 1024)}MB for ${activeTab} files`,
          variant: "destructive"
        });
        e.target.value = '';
        return;
      }

      // Set files only if validation passes
      setFiles(selectedFiles);
      
    } catch (error) {
      console.error('File selection error:', error);
      toast({
        title: "Error",
        description: "Failed to process selected files",
        variant: "destructive"
      });
      e.target.value = '';
    }
  };

  // Handle drag events
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle drop event
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      setFiles(droppedFiles);
    }
  };

  // Get accepted file types based on active tab
  const getAcceptedFileTypes = () => {
    switch (activeTab) {
      case 'image':
        return 'image/jpeg, image/png, image/gif';
      case 'video':
        return 'video/mp4, video/webm, video/quicktime';
      case 'audio':
        return 'audio/mpeg, audio/wav, audio/mp3';
      default:
        return 'image/jpeg, image/png, image/gif';
    }
  };

  // Get file type description for UI
  const getFileTypeDescription = () => {
    switch (activeTab) {
      case 'image':
        return 'JPEG, PNG or GIF files for analysis. Max file size: 10MB';
      case 'video':
        return 'MP4, WEBM or MOV video files. Max file size: 50MB';
      case 'audio':
        return 'MP3, WAV or OGG audio files. Max file size: 10MB';
      case 'webcam':
        return 'Webcam will be used for real-time analysis';
      default:
        return 'Upload files for analysis';
    }
  };

  // Get supported format text for UI
  const getSupportedFormatsText = () => {
    switch (activeTab) {
      case 'image':
        return 'Supported formats: JPEG, PNG, GIF';
      case 'video':
        return 'Supported formats: MP4, WEBM, MOV';
      case 'audio':
        return 'Supported formats: MP3, WAV, OGG';
      case 'webcam':
        return 'Ensure camera permissions are enabled';
      default:
        return 'Supported formats: Multiple formats';
    }
  };

  // Toggle webcam
  const toggleWebcam = () => {
    setIsWebcamActive(!isWebcamActive);
  };

  // Upload and analyze media
  const { mutate: uploadAndAnalyze, isPending } = useMutation({
    mutationFn: async () => {
      if (activeTab === 'webcam') {
        // Handle webcam analysis separately
        return await apiRequest('POST', '/api/analyze/webcam', {});
      }

      if (!files.length) {
        throw new Error('Please select a file to analyze');
      }

      // Validate file types
      const acceptedTypes = getAcceptedFileTypes().split(',').map(type => type.trim());
      const invalidFile = files.find(file => 
        !acceptedTypes.some(type => file.type.match(new RegExp(type.replace('*', '.*'))))
      );

      if (invalidFile) {
        throw new Error(`Invalid file type: ${invalidFile.type}`);
      }

      const formData = new FormData();
      files.forEach(file => {
        formData.append('media', file);
      });
      formData.append('type', activeTab);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || response.statusText);
      }

      return await response.json();
    },
    onSuccess(data) {
      queryClient.invalidateQueries({ queryKey: ['/api/scans/recent'] });
      navigate(`/history/${data.id}`);
    },
    onError: (error: Error) => {
      toast({
        title: "Analysis failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  return (
    <div className="bg-card rounded-xl p-6">
      <h2 className="text-xl font-poppins font-semibold text-foreground mb-4 flex items-center">
        <Upload className="text-primary mr-2" size={20} />
        Upload Media for Analysis
      </h2>
      
      {/* Upload Tabs */}
      <Tabs 
        defaultValue={activeTab} 
        value={activeTab} 
        onValueChange={handleTabChange}
        className="w-full"
      >
        <TabsList className="border-b border-muted mb-6 w-full justify-start rounded-none bg-transparent p-0 h-auto">
          <TabsTrigger 
            value="image" 
            className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            Image
          </TabsTrigger>
          <TabsTrigger 
            value="video" 
            className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            Video
          </TabsTrigger>
          <TabsTrigger 
            value="audio" 
            className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            Audio
          </TabsTrigger>
          <TabsTrigger 
            value="webcam" 
            className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            Webcam
          </TabsTrigger>
        </TabsList>

        <TabsContent value="image" className="mt-0">
          {renderUploadArea()}
        </TabsContent>
        <TabsContent value="video" className="mt-0">
          {renderUploadArea()}
        </TabsContent>
        <TabsContent value="audio" className="mt-0">
          {renderUploadArea()}
        </TabsContent>
        <TabsContent value="webcam" className="mt-0">
          {renderWebcamArea()}
        </TabsContent>
      </Tabs>
      
      {/* Upload Control Buttons */}
      <div className="mt-6 flex justify-between items-center">
        <div className="text-muted-foreground text-sm">
          <Info className="inline mr-1 text-primary" size={16} />
          <span>{getSupportedFormatsText()}</span>
        </div>
        
        <Button
          className="flex items-center gap-2"
          onClick={() => uploadAndAnalyze()}
          disabled={isPending || (activeTab !== 'webcam' && files.length === 0)}
        >
          <ScanLine size={18} />
          <span>Start Analysis</span>
        </Button>
      </div>

      {/* Progress indicator during upload/analysis */}
      {isPending && (
        <div className="mt-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-foreground">Analyzing...</span>
            <span className="text-primary">Please wait</span>
          </div>
          <Progress 
            value={50} 
            className="h-1 bg-muted" 
          />
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
        </div>
      )}
    </div>
  );
<<<<<<< HEAD
=======

  function renderUploadArea() {
    return (
      <div
        className={cn(
          "file-upload-area rounded-xl bg-card p-8 flex flex-col items-center justify-center text-center",
          dragActive && "drag-active"
        )}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        {files.length > 0 ? (
          // Files selected
          <div className="w-full">
            <div className="flex items-center justify-center mb-4">
              <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center">
                {activeTab === 'image' && <Image className="text-primary" size={32} />}
                {activeTab === 'video' && <Video className="text-primary" size={32} />}
                {activeTab === 'audio' && <Mic className="text-primary" size={32} />}
              </div>
            </div>
            
            <h3 className="font-poppins font-medium text-lg text-foreground mb-2">
              {files.length} file{files.length > 1 ? 's' : ''} selected
            </h3>
            
            <ul className="mb-4 max-w-md mx-auto text-left">
              {files.map((file, index) => (
                <li key={index} className="text-sm text-muted-foreground truncate">
                  {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                </li>
              )).slice(0, 3)}
              {files.length > 3 && (
                <li className="text-sm text-muted-foreground">
                  + {files.length - 3} more file(s)
                </li>
              )}
            </ul>
            
            <Button variant="outline" onClick={() => setFiles([])}>
              Choose Different Files
            </Button>
          </div>
        ) : (
          // No files selected
          <>
            <Upload className="text-5xl text-primary mb-4" size={48} />
            <h3 className="font-poppins font-medium text-lg text-foreground mb-2">
              Drag & Drop Files Here
            </h3>
            <p className="text-muted-foreground mb-6 max-w-md">
              Upload {getFileTypeDescription()}
            </p>
            
            <div>
              <Button onClick={handleBrowseFiles}>
                Browse Files
              </Button>
              <input
                type="file"
                className="hidden"
                accept={getAcceptedFileTypes()}
                onChange={handleFileChange}
                ref={fileInputRef}
                multiple={activeTab !== 'video'}
              />
            </div>
          </>
        )}
      </div>
    );
  }

  function renderWebcamArea() {
    return (
      <div className="rounded-xl bg-card border-2 border-primary/30 p-8 flex flex-col items-center justify-center text-center">
        <div className="relative w-full max-w-md aspect-video bg-muted rounded-lg overflow-hidden mb-6">
          {isWebcamActive ? (
            // This would be a real webcam component in a full implementation
            <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent">
              <div className="absolute top-4 right-4 h-3 w-3 rounded-full bg-accent animate-pulse"></div>
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Camera className="text-primary/40" size={64} />
            </div>
          )}
        </div>
        
        <Button 
          variant={isWebcamActive ? "destructive" : "default"} 
          onClick={toggleWebcam}
          className="mb-2"
        >
          {isWebcamActive ? "Stop Camera" : "Start Camera"}
        </Button>
        
        <p className="text-sm text-muted-foreground mt-2">
          Your webcam feed will be analyzed in real-time for deepfake detection.
          <br />No data is stored unless you explicitly save the results.
        </p>
      </div>
    );
  }
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
}
