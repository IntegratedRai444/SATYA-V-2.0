import React from 'react';
import { X } from 'lucide-react';
import { ProgressTracker } from './ProgressTracker';

interface AnalysisProgress {
  fileId: string;
  filename: string;
  jobId?: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error' | 'queued';
  message?: string;
  result?: any;
}

interface AnalysisProgressProps {
  progressItems: AnalysisProgress[];
  onRemove: (fileId: string) => void;
  onComplete?: (fileId: string, result: any) => void;
}

const ProgressItem: React.FC<{ 
  item: AnalysisProgress; 
  onRemove: (fileId: string) => void;
  onComplete?: (fileId: string, result: any) => void;
}> = ({ item, onRemove, onComplete }) => {
  const handleComplete = (result: any) => {
    if (onComplete) {
      onComplete(item.fileId, result);
    }
  };

  const handleError = (error: string) => {
    console.error(`Analysis failed for ${item.filename}:`, error);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex-1">
          <p className="text-white text-sm font-medium mb-1">{item.filename}</p>
          
          {/* Use WebSocket-based progress tracking if jobId is available */}
          {item.jobId ? (
            <ProgressTracker
              jobId={item.jobId}
              onComplete={handleComplete}
              onError={handleError}
              className="text-xs"
            />
          ) : (
            /* Fallback to legacy progress display */
            <div className="space-y-2">
              <p className="text-gray-400 text-xs">{getStatusText(item)}</p>
              
              {/* Progress Bar */}
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(item)}`}
                  style={{ width: `${item.progress}%` }}
                ></div>
              </div>
              
              {/* Progress Percentage */}
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-xs">
                  {item.status === 'error' ? 'Failed' : `${item.progress}%`}
                </span>
                {item.status === 'processing' && (
                  <span className="text-blue-400 text-xs">
                    Estimated: ~30s
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
        
        <button
          onClick={() => onRemove(item.fileId)}
          className="text-gray-400 hover:text-white transition-colors ml-3"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// Helper functions for legacy progress display
const getStatusText = (item: AnalysisProgress) => {
  switch (item.status) {
    case 'uploading':
      return 'Uploading...';
    case 'processing':
      return 'Analyzing...';
    case 'queued':
      return 'Queued for analysis...';
    case 'completed':
      return 'Analysis complete';
    case 'error':
      return item.message || 'Analysis failed';
  }
};

const getProgressColor = (item: AnalysisProgress) => {
  switch (item.status) {
    case 'completed':
      return 'bg-green-400';
    case 'error':
      return 'bg-red-400';
    case 'queued':
      return 'bg-yellow-400';
    default:
      return 'bg-blue-400';
  }
};

const AnalysisProgressComponent: React.FC<AnalysisProgressProps> = ({ 
  progressItems, 
  onRemove, 
  onComplete 
}) => {
  if (progressItems.length === 0) {
    return null;
  }

  const completedCount = progressItems.filter(item => item.status === 'completed').length;
  const errorCount = progressItems.filter(item => item.status === 'error').length;
  const queuedCount = progressItems.filter(item => item.status === 'queued').length;
  const processingCount = progressItems.filter(item => 
    item.status === 'uploading' || item.status === 'processing'
  ).length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Analysis Progress</h3>
        <div className="flex items-center space-x-4 text-sm">
          {queuedCount > 0 && (
            <span className="text-yellow-400">
              {queuedCount} queued
            </span>
          )}
          {processingCount > 0 && (
            <span className="text-blue-400">
              {processingCount} processing
            </span>
          )}
          {completedCount > 0 && (
            <span className="text-green-400">
              {completedCount} completed
            </span>
          )}
          {errorCount > 0 && (
            <span className="text-red-400">
              {errorCount} failed
            </span>
          )}
        </div>
      </div>
      
      <div className="space-y-3">
        {progressItems.map((item) => (
          <ProgressItem 
            key={item.fileId} 
            item={item} 
            onRemove={onRemove}
            onComplete={onComplete}
          />
        ))}
      </div>
      
      {completedCount > 0 && (
        <div className="text-center">
          <button className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors">
            View Analysis Results
          </button>
        </div>
      )}
    </div>
  );
};

export default AnalysisProgressComponent;