import React, { useState, useCallback } from 'react';
<<<<<<< HEAD
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Separator } from '../ui/separator';
import { useToast } from '../../hooks/use-toast';
import { Camera, FileVideo, FileAudio, Upload, Shield, Zap, Layers, Info, AlertTriangle } from "lucide-react";
import { useDropzone } from 'react-dropzone';
import AdvancedAnalysisResult from '../analysis/AdvancedAnalysisResult';
=======
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { Camera, FileVideo, FileAudio, Upload, Shield, Zap, Layers, Info, AlertTriangle } from "lucide-react";
import { useDropzone } from 'react-dropzone';
import AdvancedAnalysisResult from '@/components/analysis/AdvancedAnalysisResult';
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

interface AdvancedScanSectionProps {
  onScan?: (result: any) => void;
}

const AdvancedScanSection: React.FC<AdvancedScanSectionProps> = ({ onScan }) => {
  const { toast } = useToast();
  const [scanMode, setScanMode] = useState<'standard' | 'advanced'>('standard');
  const [activeTab, setActiveTab] = useState('image');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [selectedFiles, setSelectedFiles] = useState<{ [key: string]: File | null }>({
    image: null,
    video: null,
    audio: null
  });
  const [advancedOptions, setAdvancedOptions] = useState({
    deepAnalysis: false,
    crossCheck: true,
    preserveMetadata: true,
    sensitivity: 'medium',
  });
<<<<<<< HEAD
  const [scanError, setScanError] = useState<string | null>(null);
=======
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
  
  // For webcam
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  
  // Handle webcam activation
  const toggleWebcam = useCallback(async () => {
    if (isWebcamActive) {
      // Stop webcam
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        setWebcamStream(null);
      }
      setIsWebcamActive(false);
    } else {
      try {
        // Start webcam
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false 
        });
        setWebcamStream(stream);
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        
        setIsWebcamActive(true);
        
        toast({
          title: "Webcam activated",
          description: "Please position your face in the frame for analysis"
        });
      } catch (error) {
        console.error("Error accessing webcam:", error);
        toast({
          variant: "destructive",
          title: "Webcam Error",
          description: "Failed to access your webcam. Please check permissions."
        });
      }
    }
  }, [isWebcamActive, webcamStream, toast]);
  
  // Handle file drop for image, video or audio
  const onDrop = useCallback((acceptedFiles: File[], fileType: string) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // File size validation
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (file.size > maxSize) {
        toast({
          variant: "destructive",
          title: "File too large",
          description: "Maximum file size is 50MB"
        });
        return;
      }
      
      // Update selected file
      setSelectedFiles(prev => ({
        ...prev,
        [fileType]: file
      }));
      
      toast({
        title: "File selected",
        description: `${file.name} ready for analysis`
      });
    }
  }, [toast]);
  
  // Create dropzones for different file types
  const imageDropzone = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    maxFiles: 1,
    onDrop: (files) => onDrop(files, 'image')
  });
  
  const videoDropzone = useDropzone({
    accept: {
      'video/*': ['.mp4', '.webm', '.mov', '.avi']
    },
    maxFiles: 1,
    onDrop: (files) => onDrop(files, 'video')
  });
  
  const audioDropzone = useDropzone({
    accept: {
      'audio/*': ['.mp3', '.wav', '.ogg', '.m4a']
    },
    maxFiles: 1,
    onDrop: (files) => onDrop(files, 'audio')
  });
  
  // Take a snapshot from webcam
  const captureWebcamImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) return;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    return imageData;
  }, []);
  
  // Handle scan button click
  const handleScan = useCallback(async () => {
    setIsLoading(true);
    setAnalysisResult(null);
<<<<<<< HEAD
    setScanError(null);
=======
    
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
    try {
      let endpoint = '';
      const formData = new FormData();
      
      // Add advanced options to form data
      Object.entries(advancedOptions).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });
      
      if (activeTab === 'image') {
        endpoint = '/api/ai/analyze/image';
        
        if (isWebcamActive) {
          // Use webcam image
          const imageData = captureWebcamImage();
          if (!imageData) {
            throw new Error('Failed to capture webcam image');
          }
          formData.append('imageData', imageData);
        } else if (selectedFiles.image) {
          // Use uploaded image
          formData.append('image', selectedFiles.image);
        } else {
          throw new Error('No image selected for analysis');
        }
      } else if (activeTab === 'video') {
        endpoint = '/api/ai/analyze/video';
        
        if (!selectedFiles.video) {
          throw new Error('No video selected for analysis');
        }
        
        formData.append('video', selectedFiles.video);
      } else if (activeTab === 'audio') {
        endpoint = '/api/ai/analyze/audio';
        
        if (!selectedFiles.audio) {
          throw new Error('No audio selected for analysis');
        }
        
        formData.append('audio', selectedFiles.audio);
      } else if (activeTab === 'multimodal') {
        endpoint = '/api/ai/analyze/multimodal';
        let hasFiles = false;
        
        // Add all available files for multimodal analysis
        if (selectedFiles.image) {
          formData.append('image', selectedFiles.image);
          hasFiles = true;
        }
        
        if (selectedFiles.video) {
          formData.append('video', selectedFiles.video);
          hasFiles = true;
        }
        
        if (selectedFiles.audio) {
          formData.append('audio', selectedFiles.audio);
          hasFiles = true;
        }
        
        if (!hasFiles) {
          throw new Error('At least one file is required for multimodal analysis');
        }
      }
      
      // Make API request
<<<<<<< HEAD
      const response = await fetch(`http://localhost:5002${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let errorMsg = `Analysis failed with status: ${response.status}`;
        try {
          const errorResult = await response.json();
          if (errorResult && errorResult.message) errorMsg = errorResult.message;
        } catch {}
        setScanError(errorMsg);
        toast({
          variant: 'destructive',
          title: 'Analysis Failed',
          description: errorMsg
        });
        throw new Error(errorMsg);
      }
      const result = await response.json();
      if (!result.success) {
        setScanError(result.message || 'Analysis failed.');
        toast({
          variant: 'destructive',
          title: 'Analysis Failed',
          description: result.message || 'Analysis failed.'
        });
        throw new Error(result.message || 'Analysis failed.');
      }
      setAnalysisResult(result);
      if (onScan) {
        onScan(result);
      }
      toast({
        title: 'Analysis Complete',
        description: 'Your file was analyzed successfully.'
      });
    } catch (error) {
      if (!scanError) {
        setScanError('Failed to analyze the file. Please try again.');
        toast({
          variant: 'destructive',
          title: 'Analysis Failed',
          description: 'Failed to analyze the file. Please try again.'
        });
      }
      console.error('Scan error:', error);
    } finally {
      setIsLoading(false);
    }
  }, [activeTab, selectedFiles, advancedOptions, isWebcamActive, captureWebcamImage, onScan, toast, scanError]);
=======
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed with status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Set analysis result
      setAnalysisResult(result);
      
      // Notify parent component
      if (onScan) {
        onScan(result);
      }
      
      // Show success toast
      toast({
        title: "Analysis Complete",
        description: `Completed ${scanMode} analysis on your ${activeTab} file`
      });
    } catch (error) {
      console.error('Scan error:', error);
      
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "Something went wrong"
      });
    } finally {
      setIsLoading(false);
    }
  }, [
    activeTab,
    advancedOptions,
    captureWebcamImage,
    isWebcamActive,
    onScan,
    scanMode,
    selectedFiles,
    toast
  ]);
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
  
  // Reset selected files and results
  const handleReset = useCallback(() => {
    setSelectedFiles({
      image: null,
      video: null,
      audio: null
    });
    setAnalysisResult(null);
    
    // Stop webcam if active
    if (isWebcamActive && webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
      setIsWebcamActive(false);
    }
    
    toast({
      title: "Reset Complete",
      description: "You can now start a new analysis"
    });
  }, [isWebcamActive, webcamStream, toast]);
  
  // Toggle between standard and advanced mode
  const toggleScanMode = () => {
    setScanMode(scanMode === 'standard' ? 'advanced' : 'standard');
  };
  
  // Handle advanced option changes
  const updateAdvancedOption = (option: string, value: any) => {
    setAdvancedOptions(prev => ({
      ...prev,
      [option]: value
    }));
  };
  
  return (
    <div className="w-full space-y-6">
      {/* Mode selector */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Switch
            id="scan-mode"
            checked={scanMode === 'advanced'}
            onCheckedChange={toggleScanMode}
          />
          <Label htmlFor="scan-mode" className="text-sm font-medium">
            Advanced Mode {scanMode === 'advanced' && <Zap className="inline h-4 w-4 text-yellow-500" />}
          </Label>
        </div>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            toast({
              title: "Advanced Detection",
              description: "Advanced mode enables multimodal analysis and additional detection options for better accuracy."
            });
          }}
        >
          <Info className="h-4 w-4" />
        </Button>
      </div>
      
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="text-xl">
            {scanMode === 'advanced' ? 'Advanced AI Detection' : 'Standard Detection'}
          </CardTitle>
          <CardDescription>
            {scanMode === 'advanced' 
              ? 'Utilize our most advanced AI algorithms for deep analysis and cross-verification'
              : 'Quick detection of synthetic media using standard algorithms'}
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="image" disabled={isLoading}>
                <Camera className="h-4 w-4 mr-2" />
                Image
              </TabsTrigger>
              <TabsTrigger value="video" disabled={isLoading}>
                <FileVideo className="h-4 w-4 mr-2" />
                Video
              </TabsTrigger>
              <TabsTrigger value="audio" disabled={isLoading}>
                <FileAudio className="h-4 w-4 mr-2" />
                Audio
              </TabsTrigger>
              {scanMode === 'advanced' && (
                <TabsTrigger value="multimodal" disabled={isLoading}>
                  <Layers className="h-4 w-4 mr-2" />
                  Multimodal
                </TabsTrigger>
              )}
            </TabsList>
            
            {/* Image Tab */}
            <TabsContent value="image" className="mt-4 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Webcam section */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Live Webcam</h3>
                    <Button size="sm" variant="outline" onClick={toggleWebcam}>
                      {isWebcamActive ? 'Stop Webcam' : 'Start Webcam'}
                    </Button>
                  </div>
                  
                  <div className="relative aspect-video bg-muted rounded-md overflow-hidden flex items-center justify-center">
                    {isWebcamActive ? (
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="text-center p-4">
                        <Camera className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">Click "Start Webcam" to enable live face analysis</p>
                      </div>
                    )}
                    
                    {/* Hidden canvas for capturing frames */}
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                </div>
                
                {/* Upload section */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Upload Image</h3>
                  
                  <div 
                    {...imageDropzone.getRootProps()}
                    className={`border-2 border-dashed rounded-md p-4 text-center cursor-pointer transition-colors 
                      ${imageDropzone.isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
                      ${selectedFiles.image ? 'bg-primary/5 border-primary' : ''}
                    `}
                  >
                    <input {...imageDropzone.getInputProps()} />
                    
                    {selectedFiles.image ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-center">
                          <div className="w-16 h-16 bg-muted rounded-md overflow-hidden mr-2">
                            <img 
                              src={URL.createObjectURL(selectedFiles.image)} 
                              alt="Preview" 
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <div className="text-left">
                            <p className="text-sm font-medium truncate">{selectedFiles.image.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {(selectedFiles.image.size / (1024 * 1024)).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground">Click or drag to replace</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Upload className="h-8 w-8 mx-auto text-muted-foreground" />
                        <p className="text-sm">Drop your image here or click to browse</p>
                        <p className="text-xs text-muted-foreground">Supports JPEG, PNG, WebP (max 50MB)</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>
            
            {/* Video Tab */}
            <TabsContent value="video" className="mt-4 space-y-4">
              <div 
                {...videoDropzone.getRootProps()}
                className={`border-2 border-dashed rounded-md p-8 text-center cursor-pointer transition-colors 
                  ${videoDropzone.isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
                  ${selectedFiles.video ? 'bg-primary/5 border-primary' : ''}
                `}
              >
                <input {...videoDropzone.getInputProps()} />
                
                {selectedFiles.video ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center">
                      <FileVideo className="h-10 w-10 text-primary mr-3" />
                      <div className="text-left">
                        <p className="text-sm font-medium">{selectedFiles.video.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {(selectedFiles.video.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">Click or drag to replace video</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <FileVideo className="h-12 w-12 mx-auto text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Drop your video or click to browse</p>
                      <p className="text-xs text-muted-foreground mt-1">Supports MP4, WebM, MOV (max 50MB)</p>
                    </div>
                  </div>
                )}
              </div>
              
              {scanMode === 'advanced' && selectedFiles.video && (
                <div className="mt-4 p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded-md">
                  <div className="flex items-start">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-yellow-800 dark:text-yellow-300">Video Processing Time</p>
                      <p className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">
                        Advanced video analysis may take 1-3 minutes to complete depending on the video length and complexity.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
            
            {/* Audio Tab */}
            <TabsContent value="audio" className="mt-4 space-y-4">
              <div 
                {...audioDropzone.getRootProps()}
                className={`border-2 border-dashed rounded-md p-8 text-center cursor-pointer transition-colors 
                  ${audioDropzone.isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
                  ${selectedFiles.audio ? 'bg-primary/5 border-primary' : ''}
                `}
              >
                <input {...audioDropzone.getInputProps()} />
                
                {selectedFiles.audio ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center">
                      <FileAudio className="h-10 w-10 text-primary mr-3" />
                      <div className="text-left">
                        <p className="text-sm font-medium">{selectedFiles.audio.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {(selectedFiles.audio.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">Click or drag to replace audio</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <FileAudio className="h-12 w-12 mx-auto text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Drop your audio or click to browse</p>
                      <p className="text-xs text-muted-foreground mt-1">Supports MP3, WAV, OGG (max 50MB)</p>
                    </div>
                  </div>
                )}
              </div>
              
              {scanMode === 'advanced' && (
                <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-md">
                  <div className="flex">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-blue-800 dark:text-blue-300">Advanced Audio Analysis</p>
                      <p className="text-xs text-blue-700 dark:text-blue-400 mt-1">
                        Our AI can detect voice cloning, synthetic speech, and audio splicing with high accuracy.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
            
            {/* Multimodal Tab */}
            {scanMode === 'advanced' && (
              <TabsContent value="multimodal" className="mt-4 space-y-4">
                <div className="text-sm">
                  <p className="mb-2">Multimodal analysis combines evidence from multiple media types for more accurate detection.</p>
                  <p className="text-muted-foreground text-xs">Upload at least two media types (image, video, or audio) for multimodal analysis.</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                  {/* Image Selection for Multimodal */}
                  <div className="border rounded-md p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium">Image</h3>
                      {selectedFiles.image ? (
                        <span className="text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 px-2 py-0.5 rounded-full">
                          Selected
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">Not selected</span>
                      )}
                    </div>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="w-full"
                      onClick={() => setActiveTab('image')}
                    >
                      <Camera className="h-4 w-4 mr-2" />
                      {selectedFiles.image ? 'Change Image' : 'Add Image'}
                    </Button>
                  </div>
                  
                  {/* Video Selection for Multimodal */}
                  <div className="border rounded-md p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium">Video</h3>
                      {selectedFiles.video ? (
                        <span className="text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 px-2 py-0.5 rounded-full">
                          Selected
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">Not selected</span>
                      )}
                    </div>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="w-full"
                      onClick={() => setActiveTab('video')}
                    >
                      <FileVideo className="h-4 w-4 mr-2" />
                      {selectedFiles.video ? 'Change Video' : 'Add Video'}
                    </Button>
                  </div>
                  
                  {/* Audio Selection for Multimodal */}
                  <div className="border rounded-md p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium">Audio</h3>
                      {selectedFiles.audio ? (
                        <span className="text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300 px-2 py-0.5 rounded-full">
                          Selected
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">Not selected</span>
                      )}
                    </div>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="w-full"
                      onClick={() => setActiveTab('audio')}
                    >
                      <FileAudio className="h-4 w-4 mr-2" />
                      {selectedFiles.audio ? 'Change Audio' : 'Add Audio'}
                    </Button>
                  </div>
                </div>
                
                <div className="bg-indigo-100 dark:bg-indigo-900/30 p-3 mt-2 rounded-md">
                  <div className="flex">
                    <Shield className="h-5 w-5 text-indigo-600 dark:text-indigo-400 mr-2 flex-shrink-0" />
                    <div className="text-indigo-800 dark:text-indigo-300">
                      <p className="text-sm font-medium">Why use multimodal analysis?</p>
                      <p className="text-xs mt-1">
                        Multimodal fusion combines evidence across different media types, making it more 
                        difficult for deepfakes to evade detection. Cross-modal inconsistencies are strong 
                        indicators of manipulation.
                      </p>
                    </div>
                  </div>
                </div>
              </TabsContent>
            )}
          </Tabs>
          
          {/* Advanced options */}
          {scanMode === 'advanced' && (
            <>
              <Separator className="my-4" />
              
              <div className="space-y-4">
                <h3 className="text-sm font-medium">Advanced Options</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="deep-analysis" className="text-sm">Deep Analysis</Label>
                      <p className="text-xs text-muted-foreground">More thorough but slower analysis</p>
                    </div>
                    <Switch
                      id="deep-analysis"
                      checked={advancedOptions.deepAnalysis}
                      onCheckedChange={(checked) => updateAdvancedOption('deepAnalysis', checked)}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="cross-check" className="text-sm">Cross-Check Models</Label>
                      <p className="text-xs text-muted-foreground">Compare results from multiple detection models</p>
                    </div>
                    <Switch
                      id="cross-check"
                      checked={advancedOptions.crossCheck}
                      onCheckedChange={(checked) => updateAdvancedOption('crossCheck', checked)}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="preserve-metadata" className="text-sm">Preserve Metadata</Label>
                      <p className="text-xs text-muted-foreground">Analyze file metadata for tampering</p>
                    </div>
                    <Switch
                      id="preserve-metadata"
                      checked={advancedOptions.preserveMetadata}
                      onCheckedChange={(checked) => updateAdvancedOption('preserveMetadata', checked)}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="sensitivity" className="text-sm">Detection Sensitivity</Label>
                    <Select
                      value={advancedOptions.sensitivity}
                      onValueChange={(value) => updateAdvancedOption('sensitivity', value)}
                    >
                      <SelectTrigger id="sensitivity">
                        <SelectValue placeholder="Select sensitivity" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low (fewer false positives)</SelectItem>
                        <SelectItem value="medium">Medium (balanced)</SelectItem>
                        <SelectItem value="high">High (catch subtle manipulation)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-between pt-2">
          <Button variant="outline" onClick={handleReset} disabled={isLoading}>
            Reset
          </Button>
          
          <Button onClick={handleScan} disabled={isLoading} className="space-x-2">
            {isLoading ? (
              <>
                <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                {scanMode === 'advanced' ? <Zap className="h-4 w-4 mr-2" /> : <Shield className="h-4 w-4 mr-2" />}
                <span>Start {scanMode === 'advanced' ? 'Advanced' : 'Standard'} Scan</span>
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {/* Analysis Results */}
      {analysisResult && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
          <AdvancedAnalysisResult 
            result={analysisResult}
            onShare={() => {
              toast({
                title: "Sharing Results",
                description: "This feature will be available in the final version"
              });
            }}
            onExport={() => {
              const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(analysisResult, null, 2));
              const downloadAnchorNode = document.createElement('a');
              downloadAnchorNode.setAttribute("href", dataStr);
              downloadAnchorNode.setAttribute("download", `satya-analysis-${new Date().getTime()}.json`);
              document.body.appendChild(downloadAnchorNode);
              downloadAnchorNode.click();
              downloadAnchorNode.remove();
              
              toast({
                title: "Analysis Exported",
                description: "Results saved as JSON file"
              });
            }}
          />
        </div>
      )}
<<<<<<< HEAD
      {scanError && (
        <div className="px-6 pb-2 text-red-600 text-center font-medium">{scanError}</div>
      )}
=======
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
    </div>
  );
};

export default AdvancedScanSection;