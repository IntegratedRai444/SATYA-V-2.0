import React from 'react';
import { useNavigate } from 'react-router-dom';
import HeroSection from '../components/landing/HeroSection';
import ParticleBackground from '../components/home/ParticleBackground';
import AuthenticityScore from '../components/landing/AuthenticityScore';
import DetectionToolsPreview from '../components/landing/DetectionToolsPreview';
import ErrorBoundary from '../components/ui/ErrorBoundary';
import { Button } from '../components/ui/button';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 relative">
      {/* Particle Background */}
      <ParticleBackground />
      
      <header className="bg-transparent relative z-10">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mr-3">
                <span className="text-white text-sm font-bold">S</span>
              </div>
              <span className="text-white text-xl font-bold">Satya</span>
              <span className="text-blue-400 text-xl font-bold">AI</span>
            </div>
            <div className="flex items-center space-x-4">
              <Button 
                variant="ghost" 
                className="text-gray-300 hover:bg-gray-700 hover:text-white"
                onClick={() => navigate('/login')}
              >
                Sign In
              </Button>
              <Button 
                className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
                onClick={() => navigate('/register')}
              >
                Get Started
              </Button>
            </div>
          </div>
        </nav>
      </header>
      
      <main className="relative z-10">
        <div className="relative">
          <ErrorBoundary>
            <HeroSection />
          </ErrorBoundary>
          
          {/* Authenticity Score Display */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
            <div className="absolute top-[-200px] right-8 hidden lg:block">
              <ErrorBoundary>
                <AuthenticityScore score={97} />
              </ErrorBoundary>
            </div>
          </div>
          
          {/* Detection Tools Preview */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-white mb-4">
                Comprehensive Detection Tools
              </h2>
              <p className="text-gray-400 text-lg">
                Choose from our suite of AI-powered detection tools
              </p>
            </div>
            <div className="relative max-w-4xl mx-auto">
              <ErrorBoundary>
                <DetectionToolsPreview />
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="bg-gray-900/50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <div className="h-6 w-6 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mr-2">
                <span className="text-white text-xs font-bold">S</span>
              </div>
              <span className="text-white font-medium">SatyaAI</span>
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-400 hover:text-white">Terms</a>
              <a href="#" className="text-gray-400 hover:text-white">Privacy</a>
              <a href="#" className="text-gray-400 hover:text-white">Contact</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
