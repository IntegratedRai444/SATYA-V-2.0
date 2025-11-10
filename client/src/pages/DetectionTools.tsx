import React, { useState } from 'react';
import DetectionToolsGrid from '../components/detection/DetectionToolsGrid';
import AudioAnalyzer from '../components/realtime/AudioAnalyzer';
import NotificationBell from '../components/notifications/NotificationBell';
import { Button } from '@/components/ui/button';
import { Mic, Bell, X } from 'lucide-react';
import { useDetections } from '@/hooks/useDetections';

const DetectionTools: React.FC = () => {
  const [showAudioAnalyzer, setShowAudioAnalyzer] = useState(false);
  const [notifications, setNotifications] = useState([
    { id: 1, message: 'New security update available', read: false },
    { id: 2, message: 'Your scan is complete', read: false },
  ]);
  
  // Use detections hook for detection data
  const { detections, loading, error, filters, updateFilters, refresh } = useDetections({
    limit: 10,
    sortBy: 'newest'
  });

  const toggleAudioAnalyzer = () => setShowAudioAnalyzer(!showAudioAnalyzer);
  
  const markAsRead = (id: number) => {
    setNotifications(notifications.map(n => 
      n.id === id ? { ...n, read: true } : n
    ));
  };

  return (
    <div className="p-6 space-y-8 bg-gray-900 min-h-screen">
      {/* Header with Notifications */}
      <div className="flex justify-between items-start">
        <div className="text-center space-y-4 flex-1">
          <h1 className="text-3xl font-bold text-white">
            Choose your media type for comprehensive analysis
          </h1>
          <p className="text-gray-400 text-lg">
            Select the appropriate detection tool based on your media format
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Button 
            variant="outline" 
            size="icon"
            onClick={toggleAudioAnalyzer}
            className="bg-gray-800 hover:bg-gray-700 text-white border-gray-700"
          >
            <Mic className="h-5 w-5" />
          </Button>
          
          <NotificationBell 
            notifications={notifications}
            onNotificationClick={markAsRead}
          />
        </div>
      </div>

      {/* Audio Analyzer Overlay */}
      {showAudioAnalyzer && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-full max-w-2xl">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-white">Audio Analysis</h2>
              <Button 
                variant="ghost" 
                size="icon"
                onClick={toggleAudioAnalyzer}
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            <AudioAnalyzer />
          </div>
        </div>
      )}

      {/* Detection Tools Grid */}
      <DetectionToolsGrid />

      {/* Footer Info */}
      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-400">
          Using <span className="text-blue-400 font-medium">Neural Vision v4.2</span> models
        </div>
        <div className="text-xs text-gray-500">
          {notifications.filter(n => !n.read).length} unread notifications
        </div>
      </div>
    </div>
  );
};

export default DetectionTools;