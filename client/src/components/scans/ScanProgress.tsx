import React, { useEffect, useState, useCallback } from 'react';
import { Progress } from '@/components/ui/progress';
import { Check, Clock, AlertCircle, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useScanWebSocket } from '@/hooks/useScanWebSocket';
import { useToast } from '@/components/ui/use-toast';
import { webSocketService, WebSocketMessage } from '@/services/websocket';

interface ScanProgressProps {
  className?: string;
}

export interface ScanProgressData {
  scanId: string;
  fileName: string;
  progress: number;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  currentStep: string;
  totalSteps: number;
  currentStepIndex: number;
  timeRemaining?: number;
  error?: string;
  timestamp: string;
}

export const ScanProgress: React.FC<ScanProgressProps> = ({ className }) => {
  const [activeScans, setActiveScans] = useState<Record<string, ScanProgressData>>({});
  const [completedScans, setCompletedScans] = useState<Record<string, ScanProgressData>>({});
  const [expandedScans, setExpandedScans] = useState<Set<string>>(new Set());
  const { toast } = useToast();

  // Format time remaining
  const formatTimeRemaining = useCallback((seconds?: number) => {
    if (!seconds) return 'Calculating...';
    if (seconds < 60) return `${Math.ceil(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.ceil(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  }, []);

  // Toggle scan details
  const toggleScanDetails = useCallback((scanId: string) => {
    setExpandedScans(prev => {
      const newSet = new Set(prev);
      if (newSet.has(scanId)) {
        newSet.delete(scanId);
      } else {
        newSet.add(scanId);
      }
      return newSet;
    });
  }, []);

  // Handle scan updates from WebSocket
  const handleScanUpdate = useCallback((message: WebSocketMessage) => {
    if (message.type !== 'scan_update') return;
    
    const scanData = message as unknown as ScanProgressData;
    
    if (scanData.status === 'completed' || scanData.status === 'failed') {
      setActiveScans(prev => {
        const newScans = { ...prev };
        delete newScans[scanData.scanId];
        return newScans;
      });
      
      setCompletedScans(prev => ({
        ...prev,
        [scanData.scanId]: {
          ...scanData,
          progress: 100,
          currentStep: scanData.status === 'completed' ? 'Scan completed' : 'Scan failed',
        },
      }));
      
      // Auto-remove completed/failed scans after delay
      setTimeout(() => {
        setCompletedScans(prev => {
          const newScans = { ...prev };
          delete newScans[scanData.scanId];
          return newScans;
        });
      }, 10000);
    } else {
      setActiveScans(prev => ({
        ...prev,
        [scanData.scanId]: {
          ...scanData,
          timestamp: new Date().toISOString(),
        },
      }));
    }
  }, []);

  // Handle WebSocket errors
  const handleError = useCallback((error: Error) => {
    console.error('WebSocket error in ScanProgress:', error);
    toast({
      title: 'Connection Error',
      description: error.message || 'Failed to connect to real-time updates',
      variant: 'destructive',
      duration: 5000,
    });
    
    // Attempt to reconnect if we're not already reconnecting
    if (webSocketService.getConnectionStatus() === 'disconnected') {
      webSocketService.connect().catch(err => {
        console.error('Failed to reconnect:', err);
      });
    }
  }, [toast]);

  // Initialize WebSocket connection
  const { subscribeToScan, unsubscribeFromScan, isConnected } = useScanWebSocket({
    onScanUpdate: handleScanUpdate,
    onError: handleError,
  });

  // Subscribe to active scans
  useEffect(() => {
    const activeScanIds = Object.keys(activeScans);
    
    // Subscribe to new scans with error handling
    activeScanIds.forEach(scanId => {
      try {
        subscribeToScan(scanId);
      } catch (error) {
        console.error(`Failed to subscribe to scan ${scanId}:`, error);
      }
    });
    
    // Cleanup function to unsubscribe when component unmounts or scans change
    return () => {
      activeScanIds.forEach(scanId => {
        try {
          unsubscribeFromScan(scanId);
        } catch (error) {
          console.error(`Failed to unsubscribe from scan ${scanId}:`, error);
        }
      });
    };
  }, [activeScans, subscribeToScan, unsubscribeFromScan]);

  // Get status color
  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-500';
      case 'failed':
        return 'text-destructive';
      case 'processing':
        return 'text-blue-500';
      case 'queued':
        return 'text-amber-500';
      default:
        return 'text-muted-foreground';
    }
  }, []);

  // Get status icon
  const getStatusIcon = useCallback((status: string) => {
    switch (status) {
      case 'completed':
        return <Check className="h-4 w-4" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4" />;
      case 'processing':
        return <RefreshCw className="h-3 w-3 animate-spin" />;
      case 'queued':
        return <Clock className="h-3 w-3" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  }, []);

  // Render step indicators
  const renderStepIndicators = useCallback((scan: ScanProgressData) => {
    if (!expandedScans.has(scan.scanId)) return null;
    
    return (
      <div className="mt-2 pl-6 space-y-2 text-sm">
        {Array.from({ length: scan.totalSteps }).map((_, index) => (
          <div 
            key={index} 
            className={`flex items-center ${index < scan.currentStepIndex ? 'text-green-500' : index === scan.currentStepIndex ? 'text-blue-500 font-medium' : 'text-muted-foreground'}`}
          >
            <span className="w-4 h-4 mr-2 flex items-center justify-center">
              {index < scan.currentStepIndex ? (
                <Check className="h-3 w-3" />
              ) : index === scan.currentStepIndex ? (
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              ) : (
                <div className="w-2 h-2 rounded-full bg-muted" />
              )}
            </span>
            <span className="truncate">
              {index === scan.currentStepIndex ? scan.currentStep : `Step ${index + 1}`}
            </span>
            {index === scan.currentStepIndex && scan.timeRemaining && (
              <span className="ml-2 text-xs text-muted-foreground whitespace-nowrap">
                (ETA: {formatTimeRemaining(scan.timeRemaining)})
              </span>
            )}
          </div>
        ))}
        {scan.error && (
          <div className="text-sm text-destructive mt-2 p-2 bg-destructive/10 rounded-md">
            <p className="font-medium">Error:</p>
            <p className="text-xs">{scan.error}</p>
          </div>
        )}
      </div>
    );
  }, [expandedScans, formatTimeRemaining]);

  const allScans = Object.values({ ...activeScans, ...completedScans })
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  
  if (allScans.length === 0) return null;

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center">
            {!isConnected ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin text-amber-500" />
            ) : (
              <div className={`h-2 w-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-amber-500'}`} />
            )}
            Scan Progress
            <span className="ml-2 text-xs bg-muted text-muted-foreground rounded-full px-2 py-0.5">
              {Object.keys(activeScans).length} in progress
            </span>
          </CardTitle>
          <Button 
            variant="ghost" 
            size="sm" 
            className="h-6 text-xs text-muted-foreground"
            onClick={() => {
              // Force refresh the WebSocket connection
              webSocketService.disconnect();
              webSocketService.connect().catch(handleError);
            }}
          >
            <RefreshCw className="h-3 w-3 mr-1" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 max-h-[400px] overflow-y-auto">
        {allScans.map(scan => (
          <div key={scan.scanId} className="space-y-2 p-3 rounded-lg border">
            <div 
              className="flex items-center justify-between cursor-pointer"
              onClick={() => toggleScanDetails(scan.scanId)}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center">
                  <span className={`mr-2 ${getStatusColor(scan.status)}`}>
                    {getStatusIcon(scan.status)}
                  </span>
                  <span className="truncate font-medium">{scan.fileName}</span>
                </div>
                <div className="text-xs text-muted-foreground mt-1 flex items-center">
                  <span className="truncate">
                    {scan.status === 'processing' || scan.status === 'queued'
                      ? `${scan.currentStep} (${scan.currentStepIndex + 1}/${scan.totalSteps})`
                      : scan.status.charAt(0).toUpperCase() + scan.status.slice(1)
                    }
                  </span>
                  {scan.timeRemaining && (
                    <span className="ml-2 text-xs text-muted-foreground whitespace-nowrap">
                      • ETA: {formatTimeRemaining(scan.timeRemaining)}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center">
                <div className={`text-sm font-medium ${getStatusColor(scan.status)} mr-2`}>
                  {scan.progress}%
                </div>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-6 w-6"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleScanDetails(scan.scanId);
                  }}
                >
                  <svg
                    className={`h-4 w-4 transition-transform ${expandedScans.has(scan.scanId) ? 'rotate-180' : ''}`}
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </Button>
              </div>
            </div>
            
            <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <Progress 
                value={scan.progress} 
                className={`h-full ${scan.status === 'failed' ? 'bg-destructive' : ''}`}
              />
            </div>
            
            {expandedScans.has(scan.scanId) && (
              <div className="text-xs text-muted-foreground mt-1">
                {scan.error && (
                  <div className="text-destructive bg-destructive/10 p-2 rounded-md mb-2">
                    {scan.error}
                  </div>
                )}
                {renderStepIndicators(scan)}
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default ScanProgress;
