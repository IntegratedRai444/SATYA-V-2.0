import { useState, useEffect } from 'react';
import { AlertCircle, FileBarChart, CheckCircle, AlertTriangle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { ScanResult, DetectionDetail } from '@/lib/types';

// Props interface
interface AdvancedAnalysisCardProps {
  result?: ScanResult;
  isLoading?: boolean;
  onExport?: () => void;
}

export default function AdvancedAnalysisCard({ 
  result, 
  isLoading = false,
  onExport
}: AdvancedAnalysisCardProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [scanProgress, setScanProgress] = useState(0);
  
  // Simulate scanning progress animation when loading
  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return prev;
          }
          return prev + 5;
        });
      }, 300);
      
      return () => clearInterval(interval);
    } else {
      setScanProgress(100);
    }
  }, [isLoading]);

  if (isLoading) {
    return (
      <Card className="p-6 border border-primary/30 relative overflow-hidden">
        <div className="mb-4 flex items-center">
          <FileBarChart className="text-primary mr-2" size={20} />
          <h2 className="text-xl font-poppins font-semibold">Processing Analysis</h2>
        </div>
        
        <div className="space-y-4">
          <div className="bg-muted rounded-lg p-6 text-center relative overflow-hidden">
            <div className="absolute top-0 left-0 h-1 w-full bg-primary/20">
              <div 
                className="h-full bg-gradient-to-r from-primary to-secondary" 
                style={{ width: `${scanProgress}%`, transition: 'width 0.3s ease-in-out' }}
              ></div>
            </div>
            
            <AlertCircle className="mx-auto h-12 w-12 text-primary mb-4" />
            <h3 className="text-xl font-medium mb-2">Analyzing Media</h3>
            <p className="text-muted-foreground mb-6">
              Our AI is conducting a comprehensive analysis of your media for potential manipulation.
              This may take a few moments.
            </p>
            <div className="text-sm text-primary">
              {scanProgress < 30 ? 'Initializing analysis...' : 
               scanProgress < 60 ? 'Running deepfake detection algorithms...' :
               scanProgress < 90 ? 'Verifying results...' : 
               'Finalizing report...'}
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-20 bg-muted rounded-lg animate-pulse"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }
  
  if (!result) {
    return (
      <Card className="p-6 border border-primary/30">
        <div className="text-center py-12">
          <AlertTriangle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-xl font-medium mb-2">No Analysis Result</h3>
          <p className="text-muted-foreground">
            Please upload and analyze media to view detection results.
          </p>
        </div>
      </Card>
    );
  }
  
  const isDeepfake = result.result === 'deepfake';
  const resultColor = isDeepfake ? 'destructive' : 'accent';
  
  // Get detection details by category
  const getFaceDetails = () => result.detectionDetails?.filter(d => d.category === 'face') || [];
  const getAudioDetails = () => result.detectionDetails?.filter(d => d.category === 'audio') || [];
  const getFrameDetails = () => result.detectionDetails?.filter(d => d.category === 'frame') || [];
  const getGeneralDetails = () => result.detectionDetails?.filter(d => d.category === 'general') || [];
  
  return (
    <Card className="p-6 border border-primary/30 relative overflow-hidden">
      {/* Floating accuracy indicator */}
      <div className={cn(
        "absolute top-4 right-4 px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2",
        isDeepfake ? "bg-destructive/10 text-destructive" : "bg-accent/10 text-accent"
      )}>
        {isDeepfake ? 
          <AlertTriangle size={16} /> : 
          <CheckCircle size={16} />
        }
        <span className="uppercase">{result.result}</span>
      </div>
      
      <div className="mb-6">
        <div className="flex items-center mb-4">
          <FileBarChart className="text-primary mr-2" size={20} />
          <h2 className="text-xl font-poppins font-semibold">Advanced Analysis</h2>
        </div>
        
        <div className="p-4 rounded-lg bg-muted flex flex-col md:flex-row items-start justify-between gap-4">
          <div>
            <h3 className="font-medium text-lg mb-1">{result.filename}</h3>
            <p className="text-sm text-muted-foreground">
              {result.type === 'image' && 'Image analysis'}
              {result.type === 'video' && `Video analysis • ${result.metadata?.resolution || 'N/A'} • ${result.metadata?.duration || 'N/A'}`}
              {result.type === 'audio' && `Audio analysis • ${result.metadata?.duration || 'N/A'}`}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Analyzed on {new Date(result.timestamp).toLocaleString()}
            </p>
          </div>
          
          <div className="flex flex-col items-center">
            <div className={cn(
              "w-24 h-24 rounded-full flex items-center justify-center text-2xl font-bold relative",
              `text-${resultColor}`
            )}>
              <div className={`absolute inset-0 rounded-full border-4 border-${resultColor} opacity-20`}></div>
              <div className={`absolute inset-2 rounded-full border-2 border-${resultColor} opacity-40`}></div>
              <span>{result.confidenceScore}%</span>
            </div>
            <span className="text-sm text-muted-foreground mt-2">Confidence Score</span>
          </div>
        </div>
      </div>
      
      {/* Analysis Tabs */}
      <div className="mb-6 border-b border-muted">
        <div className="flex space-x-2 overflow-x-auto">
          <button
            className={cn(
              "px-4 py-2 text-sm font-medium",
              activeTab === 'overview' ? "border-b-2 border-primary text-primary" : "text-muted-foreground"
            )}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          {getFaceDetails().length > 0 && (
            <button
              className={cn(
                "px-4 py-2 text-sm font-medium",
                activeTab === 'face' ? "border-b-2 border-primary text-primary" : "text-muted-foreground"
              )}
              onClick={() => setActiveTab('face')}
            >
              Facial Analysis
            </button>
          )}
          {getAudioDetails().length > 0 && (
            <button
              className={cn(
                "px-4 py-2 text-sm font-medium",
                activeTab === 'audio' ? "border-b-2 border-primary text-primary" : "text-muted-foreground"
              )}
              onClick={() => setActiveTab('audio')}
            >
              Audio Analysis
            </button>
          )}
          {getFrameDetails().length > 0 && (
            <button
              className={cn(
                "px-4 py-2 text-sm font-medium",
                activeTab === 'frame' ? "border-b-2 border-primary text-primary" : "text-muted-foreground"
              )}
              onClick={() => setActiveTab('frame')}
            >
              Frame Analysis
            </button>
          )}
          <button
            className={cn(
              "px-4 py-2 text-sm font-medium",
              activeTab === 'technical' ? "border-b-2 border-primary text-primary" : "text-muted-foreground"
            )}
            onClick={() => setActiveTab('technical')}
          >
            Technical Details
          </button>
        </div>
      </div>
      
      {/* Tab Content */}
      <div className="mb-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Column - Visualization */}
            <div className="space-y-4">
              <div className="bg-muted rounded-lg p-4">
                <h4 className="font-medium text-foreground mb-3">Summary</h4>
                
                <div className="text-center py-4 mb-4">
                  {isDeepfake ? (
                    <div className="flex items-center justify-center flex-col">
                      <AlertTriangle className={`text-${resultColor} h-16 w-16 mb-2`} />
                      <h3 className={`text-${resultColor} font-bold text-2xl`}>DEEPFAKE DETECTED</h3>
                      <p className="text-muted-foreground mt-2">
                        This media shows signs of AI manipulation with {result.confidenceScore}% confidence.
                      </p>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center flex-col">
                      <CheckCircle className={`text-${resultColor} h-16 w-16 mb-2`} />
                      <h3 className={`text-${resultColor} font-bold text-2xl`}>AUTHENTIC MEDIA</h3>
                      <p className="text-muted-foreground mt-2">
                        No signs of manipulation detected with {result.confidenceScore}% confidence.
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm font-medium flex justify-between">
                    <span>Overall Confidence</span>
                    <span className={`text-${resultColor}`}>{result.confidenceScore}%</span>
                  </p>
                  <div className="w-full bg-card rounded-full h-2 overflow-hidden">
                    <div 
                      className={`bg-${resultColor} h-full`} 
                      style={{ width: `${result.confidenceScore}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Right Column - Key Findings */}
            <div className="space-y-4">
              <div className="bg-muted rounded-lg p-4">
                <h4 className="font-medium text-foreground mb-3">Key Findings</h4>
                <ul className="space-y-2">
                  {result.detectionDetails?.slice(0, 4).map((detail, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <div className={`w-6 h-6 rounded-full bg-${resultColor}/20 flex items-center justify-center flex-shrink-0 mt-0.5`}>
                        <span className={`text-${resultColor} text-xs`}>{i+1}</span>
                      </div>
                      <div>
                        <p className="font-medium">{detail.name}</p>
                        <p className="text-sm text-muted-foreground">{detail.description}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="h-1.5 flex-1 bg-card rounded-full overflow-hidden">
                            <div 
                              className={`bg-${resultColor} h-full`} 
                              style={{ width: `${detail.confidence}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-muted-foreground">{detail.confidence}%</span>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'face' && (
          <div className="bg-muted rounded-lg p-4">
            <h4 className="font-medium text-foreground mb-3">Facial Analysis Details</h4>
            {getFaceDetails().length > 0 ? (
              <div className="space-y-4">
                {getFaceDetails().map((detail, i) => (
                  <DetailItem key={i} detail={detail} />
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground italic text-center py-4">
                No facial analysis data available for this media.
              </p>
            )}
          </div>
        )}
        
        {activeTab === 'audio' && (
          <div className="bg-muted rounded-lg p-4">
            <h4 className="font-medium text-foreground mb-3">Audio Analysis Details</h4>
            {getAudioDetails().length > 0 ? (
              <div className="space-y-4">
                {getAudioDetails().map((detail, i) => (
                  <DetailItem key={i} detail={detail} />
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground italic text-center py-4">
                No audio analysis data available for this media.
              </p>
            )}
          </div>
        )}
        
        {activeTab === 'frame' && (
          <div className="bg-muted rounded-lg p-4">
            <h4 className="font-medium text-foreground mb-3">Frame Analysis Details</h4>
            {getFrameDetails().length > 0 ? (
              <div className="space-y-4">
                {getFrameDetails().map((detail, i) => (
                  <DetailItem key={i} detail={detail} />
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground italic text-center py-4">
                No frame analysis data available for this media.
              </p>
            )}
          </div>
        )}
        
        {activeTab === 'technical' && (
          <div className="bg-muted rounded-lg p-4">
            <h4 className="font-medium text-foreground mb-3">Technical Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h5 className="text-sm font-medium mb-2">Media Information</h5>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">File Type:</span>
                    <span>{result.type.toUpperCase()}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">File Size:</span>
                    <span>{result.metadata?.size || 'N/A'}</span>
                  </li>
                  {result.metadata?.resolution && (
                    <li className="flex justify-between">
                      <span className="text-muted-foreground">Resolution:</span>
                      <span>{result.metadata.resolution}</span>
                    </li>
                  )}
                  {result.metadata?.duration && (
                    <li className="flex justify-between">
                      <span className="text-muted-foreground">Duration:</span>
                      <span>{result.metadata.duration}</span>
                    </li>
                  )}
                </ul>
              </div>
              
              <div>
                <h5 className="text-sm font-medium mb-2">Analysis Information</h5>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Scan ID:</span>
                    <span>{result.id}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Analysis Date:</span>
                    <span>{new Date(result.timestamp).toLocaleString()}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Confidence Score:</span>
                    <span className={`text-${resultColor}`}>{result.confidenceScore}%</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-muted-foreground">Detection Methods:</span>
                    <span>{result.detectionDetails?.length || 0}</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

// Helper component for displaying detailed analysis items
function DetailItem({ detail }: { detail: DetectionDetail }) {
  const itemColor = detail.confidence > 70 ? 'destructive' : 'accent';
  
  return (
    <div className="p-4 bg-card rounded-lg">
      <div className="flex justify-between items-center mb-2">
        <span className="font-medium">{detail.name}</span>
        <span className={`text-${itemColor} font-semibold`}>{detail.confidence}%</span>
      </div>
      <Progress
        value={detail.confidence}
        className="h-1.5 bg-card"
      />
      <p className="mt-3 text-sm text-muted-foreground">{detail.description}</p>
    </div>
  );
}