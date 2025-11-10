import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription } from '../components/ui/alert';
import { useImageAnalysis, useVideoAnalysis, useAudioAnalysis } from '../hooks/useApi';

function Scan() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  const imageAnalysis = useImageAnalysis();
  const videoAnalysis = useVideoAnalysis();
  const audioAnalysis = useAudioAnalysis();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;

    const fileType = selectedFile.type;
    
    if (fileType.startsWith('image/')) {
      imageAnalysis.mutate(selectedFile);
    } else if (fileType.startsWith('video/')) {
      videoAnalysis.mutate(selectedFile);
    } else if (fileType.startsWith('audio/')) {
      audioAnalysis.mutate(selectedFile);
    }
  };

  const isAnalyzing = imageAnalysis.isPending || videoAnalysis.isPending || audioAnalysis.isPending;
  const analysisResult = imageAnalysis.data || videoAnalysis.data || audioAnalysis.data;

  return (
    <div className="container mx-auto p-6 space-y-6 bg-bg-primary min-h-screen">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Media Analysis</h1>
        <p className="text-text-secondary">Upload and analyze media files for deepfake detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Media</CardTitle>
            <CardDescription>Drag and drop or select files to analyze</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-primary bg-primary/10' 
                  : 'border-muted-foreground/25 hover:border-primary/50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="space-y-4">
                <div className="text-4xl">📁</div>
                <div>
                  <p className="text-lg font-medium">Drop files here</p>
                  <p className="text-sm text-muted-foreground">or click to browse</p>
                </div>
                <input
                  type="file"
                  onChange={handleFileSelect}
                  accept="image/*,video/*,audio/*"
                  className="hidden"
                  id="file-upload"
                />
                <Button 
                  variant="outline" 
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  Select File
                </Button>
              </div>
            </div>

            {selectedFile && (
              <div className="mt-4 p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{selectedFile.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>Deepfake detection results will appear here</CardDescription>
          </CardHeader>
          <CardContent>
            {isAnalyzing && (
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                <span className="ml-2">Analyzing media...</span>
              </div>
            )}

            {analysisResult && (
              <div className="space-y-4">
                <Alert variant={analysisResult.authenticity === 'AUTHENTIC MEDIA' ? 'default' : 'destructive'}>
                  <AlertDescription>
                    <div className="space-y-2">
                      <p className="font-medium">
                        Result: {analysisResult.authenticity}
                      </p>
                      <p>
                        Confidence: {analysisResult.confidence.toFixed(1)}%
                      </p>
                      <p className="text-sm">
                        Case ID: {analysisResult.case_id}
                      </p>
                    </div>
                  </AlertDescription>
                </Alert>

                {analysisResult.key_findings && (
                  <div>
                    <h4 className="font-medium mb-2">Key Findings:</h4>
                    <ul className="space-y-1 text-sm">
                      {analysisResult.key_findings.map((finding, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-primary">•</span>
                          <span>{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {!isAnalyzing && !analysisResult && (
              <div className="text-center p-8 text-muted-foreground">
                <div className="text-4xl mb-2">🔍</div>
                <p>Upload a file to start analysis</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Supported Formats */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Formats</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium mb-2">Images</h4>
              <p className="text-sm text-muted-foreground">JPG, PNG, GIF, BMP, WebP</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">Videos</h4>
              <p className="text-sm text-muted-foreground">MP4, AVI, MOV, WebM</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">Audio</h4>
              <p className="text-sm text-muted-foreground">MP3, WAV, AAC, OGG</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default Scan;