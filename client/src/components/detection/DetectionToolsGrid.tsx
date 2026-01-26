import React from 'react';
import { Image, Video, Mic } from 'lucide-react';
import DetectionToolCard from './DetectionToolCard';

interface DetectionTool {
  title: string;
  accuracy: number;
  features: string[];
  icon: React.ComponentType<{ className?: string }>;
  path: string;
}

const DetectionToolsGrid: React.FC = () => {
  const tools: DetectionTool[] = [
    {
      title: 'Image Analysis',
      accuracy: 98.2,
      features: ['Photoshop Detection', 'GAN Detection', 'Metadata Analysis'],
      icon: Image,
      path: '/image-analysis'
    },
    {
      title: 'Video Verification', 
      accuracy: 96.8,
      features: ['Facial Inconsistencies', 'Temporal Analysis', 'Lip Sync Verification'],
      icon: Video,
      path: '/video-analysis'
    },
    {
      title: 'Audio Detection',
      accuracy: 95.3,
      features: ['Voice Cloning Detection', 'Natural Patterns Analysis', 'Neural Voice Filter'],
      icon: Mic,
      path: '/audio-analysis'
    }
    // Webcam feature temporarily disabled
  ];
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {tools.map((tool) => (
        <DetectionToolCard key={tool.title} tool={tool} />
      ))}
    </div>
  );
};

export default DetectionToolsGrid;