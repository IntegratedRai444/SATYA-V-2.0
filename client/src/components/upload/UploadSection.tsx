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
      } else {
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
        </div>
      )}
    </div>
  );

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
}
