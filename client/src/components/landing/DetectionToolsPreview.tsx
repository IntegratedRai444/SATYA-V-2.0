import React from 'react';
import { Image, Video, Mic, Camera } from 'lucide-react';

const DetectionToolsPreview: React.FC = () => {
  const tools = [
    {
      title: 'Image Analysis',
      accuracy: 98.2,
      icon: Image,
      color: 'from-blue-500 to-cyan-500'
    },
    {
      title: 'Video Verification',
      accuracy: 96.8,
      icon: Video,
      color: 'from-purple-500 to-pink-500'
    },
    {
      title: 'Audio Detection',
      accuracy: 95.3,
      icon: Mic,
      color: 'from-green-500 to-emerald-500'
    },
    {
      title: 'Live Webcam',
      accuracy: 92.7,
      icon: Camera,
      color: 'from-orange-500 to-red-500'
    }
  ];

  return (
    <div className="absolute bottom-0 left-0 right-0 space-y-6">
      {/* Tools Grid */}
      <div className="grid grid-cols-2 gap-4">
        {tools.map((tool, index) => {
          const Icon = tool.icon;
          return (
            <div
              key={tool.title}
              className="bg-gray-800/60 backdrop-blur-sm rounded-lg p-4 border border-gray-700/50 hover:border-gray-600 transition-all duration-300 group cursor-pointer"
              style={{
                animationDelay: `${index * 0.1}s`
              }}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${tool.color} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium truncate">{tool.title}</p>
                  <p className="text-blue-400 text-xs font-semibold">
                    {tool.accuracy}% accuracy
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Neural Vision Badge */}
      <div className="text-center">
        <div className="inline-flex items-center px-4 py-2 bg-gray-800/80 backdrop-blur-sm rounded-full border border-gray-700/50">
          <div className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></div>
          <span className="text-gray-300 text-sm">
            Using <span className="text-blue-400 font-semibold">Neural Vision v4.2</span> models
          </span>
        </div>
      </div>
    </div>
  );
};

export default DetectionToolsPreview;