import React, { useState } from 'react';
import BatchUploader from '@/components/batch/BatchUploader';
import { useBatchProcessing } from '@/hooks/useBatchProcessing';

// TabButton Component
const TabButton: React.FC<{ 
  active: boolean; 
  onClick: () => void; 
  children: React.ReactNode 
}> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 rounded-md font-medium transition-colors ${
      active ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-100'
    }`}
  >
    {children}
  </button>
);

// MediaUpload Component
const MediaUpload: React.FC<{ 
  onFileUpload: (file: File) => void; 
  acceptedTypes: string; 
  maxSize: string; 
  activeTab: string 
}> = ({ onFileUpload, acceptedTypes, maxSize, activeTab }) => (
  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
    <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
      <span className="text-blue-600 text-2xl">
        {activeTab === 'image' ? '🖼️' : activeTab === 'video' ? '🎥' : '🎵'}
      </span>
    </div>
    <h3 className="text-lg font-medium text-gray-900 mb-2">
      Upload {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
    </h3>
    <p className="text-gray-500 mb-6">
      Drag and drop your {activeTab} file here, or click to browse
    </p>
    <input
      type="file"
      accept={acceptedTypes}
      onChange={(e) => e.target.files?.[0] && onFileUpload(e.target.files[0])}
      className="hidden"
      id="file-upload"
    />
    <label
      htmlFor="file-upload"
      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 cursor-pointer"
    >
      Select File
    </label>
    <p className="mt-2 text-xs text-gray-500">Max file size: {maxSize}</p>
  </div>
);

// AnalysisProgress Component
const AnalysisProgress: React.FC<{ 
  progressItems: Array<{id: string; name: string; progress: number}>; 
  onRemove: (id: string) => void 
}> = ({ progressItems, onRemove }) => (
  <div className="bg-white p-4 rounded-lg shadow">
    <h3 className="font-medium text-gray-900 mb-3">Upload Progress</h3>
    {progressItems.map((item) => (
      <div key={item.id} className="mb-2">
        <div className="flex justify-between text-sm mb-1">
          <span className="truncate">{item.name}</span>
          <span>{item.progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full"
            style={{ width: `${item.progress}%` }}
          ></div>
        </div>
        <button
          onClick={() => onRemove(item.id)}
          className="text-xs text-red-500 mt-1 hover:text-red-700"
        >
          Remove
        </button>
      </div>
    ))}
  </div>
);

// RecentActivity Component
const RecentActivity: React.FC = () => (
  <div className="bg-white p-4 rounded-lg shadow">
    <h3 className="font-medium text-gray-900 mb-3">Recent Activity</h3>
    <div className="space-y-3">
      <p className="text-sm text-gray-500">No recent activity</p>
    </div>
  </div>
);

// Main Component
const UploadAnalysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'image' | 'video' | 'audio' | 'webcam'>('image');
  const [analysisProgress, setAnalysisProgress] = useState<Array<{id: string; name: string; progress: number}>>([]);
  
  // Use batch processing hook for state management
  const { 
    files: batchFiles, 
    isProcessing, 
    removeFile: removeBatchFile, 
    processBatch 
  } = useBatchProcessing();

  const handleFileUpload = (file: File) => {
    const newItem = {
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      progress: 0
    };
    
    setAnalysisProgress(prev => [...prev, newItem]);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setAnalysisProgress(prev => 
        prev.map(item => 
          item.id === newItem.id 
            ? { ...item, progress: Math.min(item.progress + 10, 100) } 
            : item
        )
      );
      
      if (newItem.progress >= 100) {
        clearInterval(interval);
      }
    }, 300);
  };

  const removeProgress = (id: string) => {
    setAnalysisProgress(prev => prev.filter(item => item.id !== id));
  };

  const getAcceptedTypes = (type: string) => {
    switch (type) {
      case 'image':
        return 'image/*';
      case 'video':
        return 'video/*';
      case 'audio':
        return 'audio/*';
      default:
        return '*';
    }
  };

  const getMaxSize = (type: string) => {
    switch (type) {
      case 'image':
        return '5 MB';
      case 'video':
        return '100 MB';
      case 'audio':
        return '20 MB';
      default:
        return '10 MB';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Files</h1>
          <p className="text-gray-600">Upload your files for analysis</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Media Type Tabs */}
            <div className="flex space-x-4 justify-center lg:justify-start">
              <TabButton 
                active={activeTab === 'image'} 
                onClick={() => setActiveTab('image')}
              >
                Image
              </TabButton>
              <TabButton 
                active={activeTab === 'video'} 
                onClick={() => setActiveTab('video')}
              >
                Video
              </TabButton>
              <TabButton 
                active={activeTab === 'audio'} 
                onClick={() => setActiveTab('audio')}
              >
                Audio
              </TabButton>
              <TabButton 
                active={activeTab === 'webcam'} 
                onClick={() => setActiveTab('webcam')}
              >
                Webcam
              </TabButton>
            </div>

            {/* Upload Component */}
            {activeTab !== 'webcam' ? (
              <div className="space-y-6">
                {/* Batch Uploader with hook integration */}
                <BatchUploader />
                
                {/* Batch Processing Status */}
                {batchFiles.length > 0 && (
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="font-medium text-gray-900">Batch Processing ({batchFiles.length} files)</h3>
                      {!isProcessing && batchFiles.some(f => f.status === 'pending') && (
                        <button
                          onClick={processBatch}
                          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                          Process All
                        </button>
                      )}
                    </div>
                    <div className="space-y-2">
                      {batchFiles.map((file) => (
                        <div key={file.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                            <div className="flex items-center space-x-2 mt-1">
                              <div className="flex-1 bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all ${
                                    file.status === 'error' ? 'bg-red-500' :
                                    file.status === 'completed' ? 'bg-green-500' :
                                    'bg-blue-600'
                                  }`}
                                  style={{ width: `${file.progress}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-gray-500">{file.status}</span>
                            </div>
                          </div>
                          <button
                            onClick={() => removeBatchFile(file.id)}
                            className="ml-2 text-red-500 hover:text-red-700 text-xs"
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Single File Upload (Legacy) */}
                <MediaUpload
                  onFileUpload={handleFileUpload}
                  acceptedTypes={getAcceptedTypes(activeTab)}
                  maxSize={getMaxSize(activeTab)}
                  activeTab={activeTab}
                />
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg p-12 text-center border border-gray-700">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-white text-2xl">📹</span>
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">Webcam Analysis</h3>
                <p className="text-gray-400 mb-6">
                  Use your webcam for real-time deepfake detection
                </p>
                <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors">
                  Start Webcam Analysis
                </button>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Analysis Progress */}
            {analysisProgress.length > 0 && (
              <AnalysisProgress 
                progressItems={analysisProgress}
                onRemove={removeProgress}
              />
            )}
            
            {/* Recent Activity */}
            <RecentActivity />
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadAnalysis;