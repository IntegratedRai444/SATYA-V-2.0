import React, { useState, useEffect } from 'react';
import { X, Activity, Cpu, Server, Layers } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface SystemStatusProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SystemHealth {
  aiEngine: 'Online' | 'Offline';
  inferenceMode: 'CPU' | 'GPU';
  environment: 'Development' | 'Production';
  models: {
    image: { status: 'Available' | 'Standby' | 'Idle'; primary: string; notes: string };
    video: { status: 'Available' | 'Standby' | 'Idle'; primary: string; notes: string };
    audio: { status: 'Available' | 'Standby' | 'Idle'; primary: string; notes: string };
    text: { status: 'Available' | 'Standby' | 'Idle'; primary: string; notes: string };
  };
  runtime: {
    apiGateway: 'Ready' | 'Not Started';
    pythonService: 'Configured' | 'Connected';
    database: 'Configured' | 'Connected';
    lastAnalysis: string | null;
  };
}

const SystemStatus: React.FC<SystemStatusProps> = ({ isOpen, onClose }) => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    aiEngine: 'Online',
    inferenceMode: 'CPU',
    environment: 'Development',
    models: {
      image: { 
        status: 'Available', 
        primary: 'EfficientNet / Xception', 
        notes: 'Ready for inference' 
      },
      video: { 
        status: 'Standby', 
        primary: '3D CNN + Temporal', 
        notes: 'Activates on upload' 
      },
      audio: { 
        status: 'Available', 
        primary: 'CNN + LSTM', 
        notes: 'Voice synthesis detection' 
      },
      text: { 
        status: 'Available', 
        primary: 'NLP Transformer', 
        notes: 'AI text generation detection' 
      },
    },
    runtime: {
      apiGateway: 'Ready',
      pythonService: 'Configured',
      database: 'Configured',
      lastAnalysis: null,
    },
  });

  // Simulate real-time health monitoring
  useEffect(() => {
    if (!isOpen) return;

    const checkHealth = () => {
      // In a real implementation, this would make API calls to check actual system health
      setSystemHealth(prev => ({
        ...prev,
        runtime: {
          ...prev.runtime,
          apiGateway: Math.random() > 0.95 ? 'Not Started' : 'Ready',
          pythonService: Math.random() > 0.9 ? 'Configured' : 'Connected',
        },
      }));
    };

    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [isOpen]);

  const getStatusChip = (status: string, type: 'engine' | 'mode' | 'env' | 'runtime') => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium border transition-all duration-300";
    
    switch (type) {
      case 'engine':
        return status === 'Online' 
          ? `${baseClasses} bg-green-500/20 text-green-400 border-green-500/30 animate-pulse`
          : `${baseClasses} bg-gray-500/20 text-gray-400 border-gray-500/30`;
      case 'mode':
        return status === 'GPU'
          ? `${baseClasses} bg-purple-500/20 text-purple-400 border-purple-500/30`
          : `${baseClasses} bg-blue-500/20 text-blue-400 border-blue-500/30`;
      case 'env':
        return status === 'Production'
          ? `${baseClasses} bg-amber-500/20 text-amber-400 border-amber-500/30`
          : `${baseClasses} bg-gray-500/20 text-gray-400 border-gray-500/30`;
      case 'runtime':
        if (status === 'Ready' || status === 'Connected') {
          return `${baseClasses} bg-green-500/20 text-green-400 border-green-500/30`;
        } else if (status === 'Configured') {
          return `${baseClasses} bg-blue-500/20 text-blue-400 border-blue-500/30`;
        } else {
          return `${baseClasses} bg-gray-500/20 text-gray-400 border-gray-500/30`;
        }
      default:
        return `${baseClasses} bg-gray-500/20 text-gray-400 border-gray-500/30`;
    }
  };

  const getModelStatusDot = (status: string) => {
    switch (status) {
      case 'Available':
        return '🟢';
      case 'Standby':
        return '🟡';
      case 'Idle':
        return '🔵';
      default:
        return '⚪';
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Slide-over Panel */}
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-[#0f1419] border-l border-white/10 shadow-2xl">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-white/10">
            <div className="flex items-center gap-3">
              <Activity className="w-5 h-5 text-cyan-400" />
              <h2 className="text-xl font-bold text-white">System Status</h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Subtitle */}
            <p className="text-sm text-gray-400">
              System Initialized — Awaiting Analysis
            </p>

            {/* A. Engine Overview */}
            <Card className="bg-[#0a0f14] border border-white/10 p-6 space-y-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                Engine Overview
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">AI Engine Status</span>
                  {getStatusChip(systemHealth.aiEngine, 'engine')}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Inference Mode</span>
                  {getStatusChip(systemHealth.inferenceMode, 'mode')}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Environment</span>
                  {getStatusChip(systemHealth.environment, 'env')}
                </div>
              </div>
            </Card>

            {/* B. Model Availability Matrix */}
            <Card className="bg-[#0a0f14] border border-white/10 p-6 space-y-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-cyan-400" />
                Detection Models Availability
              </h3>
              
              <div className="space-y-3">
                {/* Header */}
                <div className="grid grid-cols-4 gap-2 text-xs text-gray-500 font-medium">
                  <span>Modality</span>
                  <span>Primary Model</span>
                  <span>Status</span>
                  <span>Notes</span>
                </div>

                {/* Image Model */}
                <div className="grid grid-cols-4 gap-2 items-center text-xs">
                  <span className="text-gray-300">Image</span>
                  <span className="text-gray-400 truncate" title={systemHealth.models.image.primary}>
                    {systemHealth.models.image.primary}
                  </span>
                  <div className="flex items-center gap-1">
                    <span>{getModelStatusDot(systemHealth.models.image.status)}</span>
                    <span className="text-gray-400">{systemHealth.models.image.status}</span>
                  </div>
                  <span className="text-gray-500 truncate" title={systemHealth.models.image.notes}>
                    {systemHealth.models.image.notes}
                  </span>
                </div>

                {/* Video Model */}
                <div className="grid grid-cols-4 gap-2 items-center text-xs">
                  <span className="text-gray-300">Video</span>
                  <span className="text-gray-400 truncate" title={systemHealth.models.video.primary}>
                    {systemHealth.models.video.primary}
                  </span>
                  <div className="flex items-center gap-1">
                    <span>{getModelStatusDot(systemHealth.models.video.status)}</span>
                    <span className="text-gray-400">{systemHealth.models.video.status}</span>
                  </div>
                  <span className="text-gray-500 truncate" title={systemHealth.models.video.notes}>
                    {systemHealth.models.video.notes}
                  </span>
                </div>

                {/* Audio Model */}
                <div className="grid grid-cols-4 gap-2 items-center text-xs">
                  <span className="text-gray-300">Audio</span>
                  <span className="text-gray-400 truncate" title={systemHealth.models.audio.primary}>
                    {systemHealth.models.audio.primary}
                  </span>
                  <div className="flex items-center gap-1">
                    <span>{getModelStatusDot(systemHealth.models.audio.status)}</span>
                    <span className="text-gray-400">{systemHealth.models.audio.status}</span>
                  </div>
                  <span className="text-gray-500 truncate" title={systemHealth.models.audio.notes}>
                    {systemHealth.models.audio.notes}
                  </span>
                </div>

                {/* Text Model */}
                <div className="grid grid-cols-4 gap-2 items-center text-xs">
                  <span className="text-gray-300">Text</span>
                  <span className="text-gray-400 truncate" title={systemHealth.models.text.primary}>
                    {systemHealth.models.text.primary}
                  </span>
                  <div className="flex items-center gap-1">
                    <span>{getModelStatusDot(systemHealth.models.text.status)}</span>
                    <span className="text-gray-400">{systemHealth.models.text.status}</span>
                  </div>
                  <span className="text-gray-500 truncate" title={systemHealth.models.text.notes}>
                    {systemHealth.models.text.notes}
                  </span>
                </div>
              </div>

              {/* Tooltip hint */}
              <div className="pt-2 border-t border-white/5">
                <p className="text-xs text-gray-500 italic">
                  💡 Models activate automatically when relevant media is uploaded
                </p>
              </div>
            </Card>

            {/* C. Runtime Connectivity */}
            <Card className="bg-[#0a0f14] border border-white/10 p-6 space-y-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Server className="w-5 h-5 text-cyan-400" />
                Runtime Connectivity
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">API Gateway</span>
                  {getStatusChip(systemHealth.runtime.apiGateway, 'runtime')}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Python ML Service</span>
                  {getStatusChip(systemHealth.runtime.pythonService, 'runtime')}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Database</span>
                  {getStatusChip(systemHealth.runtime.database, 'runtime')}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Last Analysis</span>
                  <span className="text-sm font-medium text-gray-400">
                    {systemHealth.runtime.lastAnalysis || 'Not started yet'}
                  </span>
                </div>
              </div>
            </Card>

            {/* Footer Note */}
            <div className="pt-4 border-t border-white/10">
              <p className="text-xs text-gray-500 text-center">
                Real metrics will appear once analysis begins
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;
