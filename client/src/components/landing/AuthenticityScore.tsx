import React from 'react';
import { Shield, Zap, CheckCircle } from 'lucide-react';

interface AuthenticityScoreProps {
  score: number;
}

const AuthenticityScore: React.FC<AuthenticityScoreProps> = ({ score }) => {
  return (
    <div className="absolute top-0 right-0 bg-gray-800/90 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 shadow-2xl max-w-sm">
      <div className="text-center">
        {/* Score Display */}
        <div className="mb-6">
          <div className="text-sm text-blue-400 uppercase tracking-widest font-semibold mb-3">
            AUTHENTICITY SCORE
          </div>
          <div className="relative">
            <div className="text-5xl font-bold text-blue-400 mb-2">{score}%</div>
            {/* Animated Ring */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-20 h-20 border-4 border-blue-400/20 rounded-full"></div>
              <div 
                className="absolute w-20 h-20 border-4 border-blue-400 rounded-full border-t-transparent animate-spin"
                style={{ animationDuration: '3s' }}
              ></div>
            </div>
          </div>
        </div>
        
        {/* Features List */}
        <div className="space-y-4">
          <div className="flex items-center text-sm text-gray-300 group">
            <div className="w-2 h-2 bg-blue-400 rounded-full mr-3 group-hover:scale-125 transition-transform"></div>
            <Zap className="w-4 h-4 text-blue-400 mr-2" />
            <span>Real-time Analysis</span>
          </div>
          <div className="flex items-center text-sm text-gray-300 group">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-3 group-hover:scale-125 transition-transform"></div>
            <Shield className="w-4 h-4 text-green-400 mr-2" />
            <span>Secure Processing</span>
          </div>
          <div className="flex items-center text-sm text-gray-300 group">
            <div className="w-2 h-2 bg-blue-400 rounded-full mr-3 group-hover:scale-125 transition-transform"></div>
            <CheckCircle className="w-4 h-4 text-blue-400 mr-2" />
            <span>Verified Protection</span>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="mt-6 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-green-400 font-medium">SYSTEM ACTIVE</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthenticityScore;