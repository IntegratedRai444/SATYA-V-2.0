import React from 'react';
import { useLocation } from 'wouter';
import { Sparkles, Upload, Play } from 'lucide-react';
import AuthenticityScore from './AuthenticityScore';
import DetectionToolsPreview from './DetectionToolsPreview';

const HeroSection: React.FC = () => {
  const [, setLocation] = useLocation();

  const handleAnalyzeMedia = () => {
    setLocation('/upload');
  };

  const handleHowItWorks = () => {
    setLocation('/how-it-works');
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 right-20 w-32 h-32 border border-blue-400 rounded-full"></div>
        <div className="absolute bottom-20 left-20 w-24 h-24 border border-blue-400 rounded-full"></div>
        <div className="absolute top-1/2 left-1/3 w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
        <div className="absolute top-1/4 right-1/4 w-1 h-1 bg-blue-400 rounded-full animate-pulse delay-1000"></div>
        <div className="absolute bottom-1/3 right-1/3 w-1 h-1 bg-blue-400 rounded-full animate-pulse delay-2000"></div>
      </div>
      
      <div className="relative z-10 container mx-auto px-6 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-screen">
          {/* Left Side - Main Content */}
          <div className="space-y-8">
            {/* Badge */}
            <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-full border border-blue-500/30 backdrop-blur-sm">
              <Sparkles className="w-4 h-4 text-blue-400 mr-2" />
              <span className="text-blue-300 text-sm font-medium">New AI Models Released</span>
              <span className="ml-2 px-2 py-1 bg-blue-600 text-white text-xs rounded-full">Protection</span>
            </div>
            
            {/* Main Headline */}
            <h1 className="text-5xl lg:text-6xl font-bold text-white leading-tight">
              Detect <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">deepfakes</span> with the<br />
              power of SatyaAI
            </h1>
            
            {/* Description */}
            <p className="text-xl text-gray-300 leading-relaxed max-w-2xl">
              Our advanced detection system helps you authenticate media with unprecedented 
              accuracy, exposing manipulated content across images, videos, and audio.
            </p>
            
            {/* Call to Action Text */}
            <p className="text-gray-400 leading-relaxed max-w-2xl">
              Upload your files or use your webcam for real-time analysis and get detailed 
              authenticity reports instantly.
            </p>
            
            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <button 
                onClick={handleAnalyzeMedia}
                className="group bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center shadow-lg hover:shadow-blue-500/25 hover:scale-105"
              >
                <Upload className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                Analyze Media
              </button>
              <button 
                onClick={handleHowItWorks}
                className="group border-2 border-gray-600 hover:border-blue-500 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center hover:bg-blue-500/10"
              >
                <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                How it Works
              </button>
            </div>
          </div>
          
          {/* Right Side - Interactive Elements */}
          <div className="relative">
            <AuthenticityScore score={89} />
            <DetectionToolsPreview />
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;