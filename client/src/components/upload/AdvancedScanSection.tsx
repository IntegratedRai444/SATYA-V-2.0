import { useState, useEffect, useRef } from "react";
import { 
  UploadCloud, 
  Image as ImageIcon, 
  Video,
  Mic, 
  Camera, 
  Info, 
  ScanLine, 
  Zap,
  Activity,
  FileCheck
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { apiRequest } from "@/lib/queryClient";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { useLocation } from "wouter";
import { useNavigation } from "@/hooks/useNavigation";
import { Badge } from "@/components/ui/badge";

export default function AdvancedScanSection() {
  const [location] = useLocation();
  const { navigate } = useNavigation();
  const { toast } = useToast();
  
  // Get the active tab from URL query parameter
  const searchParams = new URLSearchParams(location.split("?")[1] || "");
  const defaultType = searchParams.get("type") || "image";
  
  const [activeTab, setActiveTab] = useState(defaultType);
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);

  // Update URL when tab changes
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    navigate(`/scan?type=${value}&advanced=true`);
  };

  // Handle file upload via browse button
  const handleBrowseFiles = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(selectedFiles);
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
      case 'multimodal':
        return 'Upload multiple file types for enhanced analysis';
      default:
        return 'Upload files for analysis';
    }
  };

  // Toggle webcam
  const toggleWebcam = () => {
    if (!isWebcamActive) {
      // Start webcam
      if (navigator.mediaDevices?.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            if (webcamRef.current) {
              webcamRef.current.srcObject = stream;
            }
            setIsWebcamActive(true);
          })
          .catch(error => {
            console.error('Error accessing webcam:', error);
            toast({
              title: "Webcam Error",
              description: "Unable to access webcam. Please check permissions.",
              variant: "destructive"
            });
          });
      }
    } else {
      // Stop webcam
      if (webcamRef.current && webcamRef.current.srcObject) {
        const stream = webcamRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        webcamRef.current.srcObject = null;
        setIsWebcamActive(false);
      }
    }
  };

  // Clean up webcam on unmount
  useEffect(() => {
    return () => {
      if (webcamRef.current && webcamRef.current.srcObject) {
        const stream = webcamRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  // Upload and analyze media
  const { mutate: uploadAndAnalyze, isPending } = useMutation({
    mutationFn: async () => {
      if (activeTab === 'webcam') {
        // Handle webcam analysis 
        if (!webcamRef.current || !isWebcamActive) {
          throw new Error('Please start the webcam first');
        }
        
        // Capture current webcam frame
        const canvas = document.createElement('canvas');
        canvas.width = webcamRef.current.videoWidth;
        canvas.height = webcamRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(webcamRef.current, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Send to server
        const response = await apiRequest('POST', '/api/analyze/webcam', { imageData });
        return response;
      } else if (activeTab === 'multimodal') {
        // Handle multimodal analysis with potentially multiple file types
        if (!files.length) {
          throw new Error('Please select files to analyze');
        }
        
        const formData = new FormData();
        files.forEach(file => {
          // Categorize file based on type
          if (file.type.startsWith('image/')) {
            formData.append('image', file);
          } else if (file.type.startsWith('video/')) {
            formData.append('video', file);
          } else if (file.type.startsWith('audio/')) {
            formData.append('audio', file);
          }
        });
        
        const response = await fetch('/api/analyze/multimodal', {
          method: 'POST',
          body: formData,
          credentials: 'include'
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || response.statusText);
        }
        
        return await response.json();
      } else {
        // Standard single media type analysis
        if (!files.length) {
          throw new Error('Please select a file to analyze');
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
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/scans/recent'] });
      navigate(`/history/${data.id}`);
    },
    onError: (error: Error) => {
      toast({
        title: "Analysis failed",
        description: error.message,
        variant: "destructive"
      });
      setScanProgress(0);
    }
  });
  
  // Update progress during analysis
  useEffect(() => {
    if (isPending) {
      const interval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return prev;
          }
          return prev + Math.random() * 10;
        });
      }, 500);
      
      return () => clearInterval(interval);
    } else {
      setScanProgress(0);
    }
  }, [isPending]);

  return (
    <div className="bg-card rounded-xl p-6 border border-primary/30 relative overflow-hidden">
      {/* Background visual effect */}
      <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-primary/5 to-transparent rounded-full blur-3xl"></div>
      <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-secondary/5 to-transparent rounded-full blur-3xl"></div>
      
      <h2 className="text-xl font-poppins font-semibold text-foreground mb-4 flex items-center relative z-10">
        <Zap className="text-primary mr-2 animate-pulse-glow" size={20} />
        Advanced Satya Analysis
      </h2>
      
      <div className="relative z-10">
        {/* Upload Tabs */}
        <Tabs 
          defaultValue={activeTab} 
          value={activeTab} 
          onValueChange={handleTabChange}
          className="w-full"
        >
          <TabsList className="border-b border-muted mb-6 w-full justify-start rounded-none bg-transparent p-0 h-auto overflow-x-auto">
            <TabsTrigger 
              value="image" 
              className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
            >
              <ImageIcon className="mr-2 h-4 w-4" />
              Image
            </TabsTrigger>
            <TabsTrigger 
              value="video" 
              className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
            >
              <Video className="mr-2 h-4 w-4" />
              Video
            </TabsTrigger>
            <TabsTrigger 
              value="audio" 
              className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
            >
              <Mic className="mr-2 h-4 w-4" />
              Audio
            </TabsTrigger>
            <TabsTrigger 
              value="webcam" 
              className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
            >
              <Camera className="mr-2 h-4 w-4" />
              Webcam
            </TabsTrigger>
            <TabsTrigger 
              value="multimodal" 
              className="rounded-none border-b-2 border-transparent px-4 py-3 data-[state=active]:border-secondary data-[state=active]:bg-transparent data-[state=active]:text-secondary data-[state=active]:shadow-none"
            >
              <Activity className="mr-2 h-4 w-4" />
              Multimodal
              <Badge variant="outline" className="ml-2 bg-secondary/10 text-secondary text-xs">Advanced</Badge>
            </TabsTrigger>
          </TabsList>

          {/* Tab Content */}
          <TabsContent value="image" className="mt-0 relative min-h-[350px]">
            {renderFileUploadArea()}
          </TabsContent>
          <TabsContent value="video" className="mt-0 relative min-h-[350px]">
            {renderFileUploadArea()}
          </TabsContent>
          <TabsContent value="audio" className="mt-0 relative min-h-[350px]">
            {renderFileUploadArea()}
          </TabsContent>
          <TabsContent value="webcam" className="mt-0 relative min-h-[350px]">
            {renderWebcamArea()}
          </TabsContent>
          <TabsContent value="multimodal" className="mt-0 relative min-h-[350px]">
            {renderMultimodalArea()}
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Progress indicator during upload/analysis */}
      {isPending && (
        <div className="mt-4 relative z-10">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-foreground">{getAnalysisStatusMessage()}</span>
            <span className="text-primary">{Math.min(Math.round(scanProgress), 99)}%</span>
          </div>
          <div className="h-1.5 w-full bg-muted/30 rounded-full overflow-hidden relative">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary absolute top-0 left-0 rounded-full transition-all duration-300"
              style={{ width: `${scanProgress}%` }}
            ></div>
            <div className="scanner-line"></div>
          </div>
        </div>
      )}
      
      {/* Upload Control Buttons */}
      <div className="mt-6 flex justify-between items-center relative z-10">
        <div className="text-muted-foreground text-sm">
          <Info className="inline mr-1 text-primary" size={16} />
          {getFileTypeDescription()}
        </div>
        
        <Button
          className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90 shadow-[0_0_15px_rgba(0,200,255,0.5)] flex items-center gap-2 text-black font-semibold"
          onClick={() => uploadAndAnalyze()}
          disabled={isPending || 
            (activeTab === 'webcam' && !isWebcamActive) || 
            (activeTab !== 'webcam' && files.length === 0)}
        >
          <ScanLine size={18} />
          <span>Start Advanced Analysis</span>
        </Button>
      </div>
    </div>
  );

  function renderFileUploadArea() {
    return (
      <div
        className={cn(
          "file-upload-area rounded-xl bg-muted/50 p-8 flex flex-col items-center justify-center text-center",
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
              <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center pulse-glow">
                {activeTab === 'image' && <ImageIcon className="text-primary" size={32} />}
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
                  <FileCheck className="inline-block text-accent mr-2" size={14} />
                  {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                </li>
              )).slice(0, 3)}
              {files.length > 3 && (
                <li className="text-sm text-muted-foreground">
                  + {files.length - 3} more file(s)
                </li>
              )}
            </ul>
            
            <Button 
              variant="outline" 
              onClick={() => setFiles([])}
              className="border-primary/30 text-primary hover:bg-primary/10"
            >
              Choose Different Files
            </Button>
          </div>
        ) : (
          // No files selected
          <>
            <UploadCloud className="text-5xl text-primary mb-4 animate-pulse-glow" size={64} />
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
      <div className="rounded-xl bg-muted/50 border-2 border-primary/30 p-8 flex flex-col items-center justify-center text-center">
        <div className="relative w-full max-w-md aspect-video bg-muted/80 rounded-lg overflow-hidden mb-6">
          {isWebcamActive ? (
            <div className="absolute inset-0">
              <video 
                ref={webcamRef} 
                autoPlay 
                playsInline 
                className="h-full w-full object-cover"
              />
              <div className="absolute inset-0 border border-primary/30 pointer-events-none"></div>
              <div className="absolute top-2 right-2 h-3 w-3 rounded-full bg-accent animate-ping"></div>
              <div className="absolute top-2 right-2 h-3 w-3 rounded-full bg-accent"></div>
              
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3">
                <div className="text-xs text-white opacity-80 flex items-center">
                  <Activity className="inline-block mr-1" size={12} />
                  Satyalive™ Facial Analysis Active
                </div>
              </div>
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Camera className="text-primary/40" size={64} />
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3">
                <div className="text-xs text-white opacity-80">Click Start Camera to begin</div>
              </div>
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

  function renderMultimodalArea() {
    return (
      <div
        className={cn(
          "file-upload-area rounded-xl bg-gradient-to-br from-secondary/10 to-primary/10 p-8 flex flex-col items-center justify-center text-center",
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
            <div className="flex items-center justify-center mb-4 gap-2">
              <div className="h-14 w-14 rounded-full bg-primary/20 flex items-center justify-center">
                <Activity className="text-primary animate-pulse" size={28} />
              </div>
            </div>
            
            <h3 className="font-poppins font-medium text-lg text-foreground mb-2">
              {files.length} file{files.length > 1 ? 's' : ''} selected for multimodal analysis
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4 max-w-2xl mx-auto">
              <div className="bg-muted/70 rounded-lg p-3">
                <h4 className="font-medium text-sm text-foreground mb-1">Images</h4>
                <p className="text-xs text-muted-foreground">
                  {files.filter(f => f.type.startsWith('image/')).length} files
                </p>
              </div>
              <div className="bg-muted/70 rounded-lg p-3">
                <h4 className="font-medium text-sm text-foreground mb-1">Videos</h4>
                <p className="text-xs text-muted-foreground">
                  {files.filter(f => f.type.startsWith('video/')).length} files
                </p>
              </div>
              <div className="bg-muted/70 rounded-lg p-3">
                <h4 className="font-medium text-sm text-foreground mb-1">Audio</h4>
                <p className="text-xs text-muted-foreground">
                  {files.filter(f => f.type.startsWith('audio/')).length} files
                </p>
              </div>
            </div>
            
            <Button 
              variant="outline" 
              onClick={() => setFiles([])}
              className="border-secondary/30 text-secondary hover:bg-secondary/10"
            >
              Choose Different Files
            </Button>
          </div>
        ) : (
          // No files selected
          <>
            <Activity className="text-5xl text-secondary mb-4" size={64} />
            <h3 className="font-poppins font-medium text-lg text-foreground mb-2">
              Advanced Multimodal Analysis
            </h3>
            <p className="text-muted-foreground mb-6 max-w-md">
              Upload multiple file types (images, videos, audio) for enhanced cross-modal detection capabilities.
              Our advanced algorithm will perform correlation analysis across different modalities.
            </p>
            
            <div>
              <Button 
                onClick={handleBrowseFiles}
                className="bg-secondary hover:bg-secondary/90 text-black"
              >
                Browse Multiple Files
              </Button>
              <input
                type="file"
                className="hidden"
                accept="image/*, video/*, audio/*"
                onChange={handleFileChange}
                ref={fileInputRef}
                multiple={true}
              />
            </div>
          </>
        )}
      </div>
    );
  }

  function getAnalysisStatusMessage() {
    if (scanProgress < 25) {
      return "Initializing analysis algorithms...";
    } else if (scanProgress < 50) {
      return "Extracting media features...";
    } else if (scanProgress < 75) {
      return "Running neural network detection...";
    } else {
      return "Finalizing analysis report...";
    }
  }
}