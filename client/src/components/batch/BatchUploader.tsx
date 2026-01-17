import React, { useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { File, Upload, X, Check, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useBatchProcessing, type BatchFile } from '@/hooks/useBatchProcessing';
import { formatFileSize } from '@/lib/utils';
import { cn } from '@/lib/utils';

const BatchUploader: React.FC = () => {
  const { files, addFiles, removeFile, processBatch, isProcessing } = useBatchProcessing();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      addFiles(acceptedFiles);
    },
    [addFiles]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif'],
      'video/*': ['.mp4', '.webm', '.mov'],
      'audio/*': ['.mp3', '.wav', '.ogg'],
    },
    maxFiles: 20,
    maxSize: 1024 * 1024 * 500, // 500MB
    disabled: isProcessing,
  });

  const handleProcessBatch = async () => {
    try {
      await processBatch();
    } catch (error) {
      console.error('Error processing batch:', error);
    }
  };

  const getStatusIcon = (status: BatchFile['status']) => {
    switch (status) {
      case 'completed':
        return <Check className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'processing':
      case 'uploading':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      default:
        return <File className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusText = (status: BatchFile['status']) => {
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  return (
    <div className="space-y-6">
      <Card className="border-dashed border-2">
        <div
          {...getRootProps()}
          className={`p-8 text-center cursor-pointer transition-colors ${
            isDragActive ? 'bg-accent/50' : 'hover:bg-accent/20'
          }`}
        >
          <input {...getInputProps()} ref={fileInputRef} />
          <div className="flex flex-col items-center justify-center space-y-2">
            <Upload className="w-12 h-12 text-muted-foreground" />
            <div className="text-lg font-medium">
              {isDragActive ? 'Drop files here' : 'Drag & drop files here, or click to select'}
            </div>
            <p className="text-sm text-muted-foreground">
              Supports images, videos, and audio files (max 500MB per file)
            </p>
            <Button type="button" variant="outline" className="mt-2" disabled={isProcessing}>
              Select Files
            </Button>
          </div>
        </div>
      </Card>

      {files.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Batch Queue</CardTitle>
                <CardDescription>
                  {files.filter(f => f.status === 'completed').length} of {files.length} files processed
                </CardDescription>
              </div>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isProcessing}
                >
                  Add More
                </Button>
                <Button
                  onClick={handleProcessBatch}
                  disabled={isProcessing || files.every(f => f.status !== 'pending')}
                >
                  {isProcessing ? 'Processing...' : 'Process All'}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {files.map((file) => (
                <div key={file.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(file.status)}
                      <div className="font-medium truncate max-w-xs">{file.name}</div>
                      <span className="text-sm text-muted-foreground">
                        {formatFileSize(file.size)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-muted-foreground">
                        {getStatusText(file.status)}
                      </span>
                      {file.status === 'pending' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFile(file.id);
                          }}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                  {file.status !== 'pending' && file.status !== 'completed' && (
                    <div className="mt-2">
                      <Progress value={file.progress} className="h-2" />
                      {file.error && (
                        <p className="mt-1 text-sm text-red-500">{file.error}</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BatchUploader;
