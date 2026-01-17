import React from 'react';
import { Progress } from '../ui/progress';
import { CheckCircle, XCircle, Clock, AlertCircle, Loader2 } from 'lucide-react';
import { useJobProgress } from '../../hooks/useWebSocket';

interface ProgressTrackerProps {
  jobId: string | null;
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
  className?: string;
}

export function ProgressTracker({ 
  jobId, 
  onComplete, 
  onError, 
  className = '' 
}: ProgressTrackerProps) {
  const { 
    progress, 
    isConnected, 
    isCompleted, 
    isFailed, 
    isProcessing, 
    isQueued,
    isCancelled 
  } = useJobProgress(jobId);

  // Handle completion
  React.useEffect(() => {
    if (isCompleted && progress?.result && onComplete) {
      onComplete(progress.result);
    }
  }, [isCompleted, progress?.result, onComplete]);

  // Handle errors
  React.useEffect(() => {
    if (isFailed && progress?.error && onError) {
      onError(progress.error);
    }
  }, [isFailed, progress?.error, onError]);

  if (!jobId) {
    return null;
  }

  if (!isConnected) {
    return (
      <div className={`flex items-center space-x-2 text-yellow-600 ${className}`}>
        <AlertCircle className="h-4 w-4" />
        <span className="text-sm">Connecting to progress updates...</span>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className={`flex items-center space-x-2 text-gray-500 ${className}`}>
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-sm">Loading progress...</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    if (isCompleted) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
    if (isFailed) {
      return <XCircle className="h-5 w-5 text-red-500" />;
    }
    if (isCancelled) {
      return <XCircle className="h-5 w-5 text-gray-500" />;
    }
    if (isProcessing) {
      return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
    }
    if (isQueued) {
      return <Clock className="h-5 w-5 text-yellow-500" />;
    }
    return null;
  };

  const getStatusText = () => {
    if (isCompleted) return 'Analysis Complete';
    if (isFailed) return 'Analysis Failed';
    if (isCancelled) return 'Analysis Cancelled';
    if (isProcessing) return 'Analyzing...';
    if (isQueued) return 'Queued for Analysis';
    return 'Unknown Status';
  };

  const getStatusColor = () => {
    if (isCompleted) return 'text-green-600';
    if (isFailed) return 'text-red-600';
    if (isCancelled) return 'text-gray-600';
    if (isProcessing) return 'text-blue-600';
    if (isQueued) return 'text-yellow-600';
    return 'text-gray-600';
  };

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getElapsedTime = () => {
    if (!progress?.startTime) return null;
    
    const startTime = new Date(progress.startTime);
    const endTime = progress.endTime ? new Date(progress.endTime) : new Date();
    const elapsed = (endTime.getTime() - startTime.getTime()) / 1000;
    
    return elapsed;
  };

  const elapsedTime = getElapsedTime();

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <span className={`font-medium ${getStatusColor()}`}>
            {getStatusText()}
          </span>
        </div>
        
        {elapsedTime && (
          <span className="text-sm text-gray-500">
            {formatTime(elapsedTime)}
            {progress.estimatedTimeRemaining && isProcessing && (
              <span className="ml-1">
                (~{formatTime(progress.estimatedTimeRemaining)} remaining)
              </span>
            )}
          </span>
        )}
      </div>

      {/* Progress Bar */}
      {(isProcessing || isQueued) && (
        <div className="space-y-2">
          <Progress 
            value={progress.percentage} 
            className="h-2"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>{progress.stage}</span>
            <span>{progress.percentage}%</span>
          </div>
        </div>
      )}

      {/* Progress Message */}
      {progress.message && (
        <p className="text-sm text-gray-600">
          {progress.message}
        </p>
      )}

      {/* Error Message */}
      {isFailed && progress.error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3">
          <div className="flex items-center space-x-2">
            <XCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm font-medium text-red-800">Analysis Failed</span>
          </div>
          <p className="text-sm text-red-700 mt-1">{progress.error}</p>
        </div>
      )}

      {/* Success Message */}
      {isCompleted && (
        <div className="bg-green-50 border border-green-200 rounded-md p-3">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-4 w-4 text-green-500" />
            <span className="text-sm font-medium text-green-800">
              Analysis completed successfully
            </span>
          </div>
          {elapsedTime && (
            <p className="text-sm text-green-700 mt-1">
              Completed in {formatTime(elapsedTime)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// Simplified progress indicator for inline use
export function ProgressIndicator({ 
  jobId, 
  size = 'sm' 
}: { 
  jobId: string | null; 
  size?: 'sm' | 'md' | 'lg' 
}) {
  const { progress, isProcessing, isQueued } = useJobProgress(jobId);

  if (!jobId || !progress || (!isProcessing && !isQueued)) {
    return null;
  }

  const sizeClasses = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5'
  };

  return (
    <div className="flex items-center space-x-2">
      <Loader2 className={`${sizeClasses[size]} animate-spin text-blue-500`} />
      <span className="text-xs text-gray-500">
        {progress.percentage}%
      </span>
    </div>
  );
}