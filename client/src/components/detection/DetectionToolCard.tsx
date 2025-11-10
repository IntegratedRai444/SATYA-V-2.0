import React from 'react';
import { Check, ArrowRight } from 'lucide-react';
import { useLocation } from 'wouter';

interface DetectionTool {
  title: string;
  accuracy: number;
  features: string[];
  icon: React.ComponentType<{ className?: string }>;
  path: string;
}

interface DetectionToolCardProps {
  tool: DetectionTool;
}

const DetectionToolCard: React.FC<DetectionToolCardProps> = ({ tool }) => {
  const [, setLocation] = useLocation();

  const handleStartAnalysis = () => {
    setLocation(tool.path);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-all duration-300 group">
      <div className="flex items-center justify-between mb-4">
        <tool.icon className="w-8 h-8 text-blue-400 group-hover:scale-110 transition-transform" />
        <span className="text-sm text-blue-400 font-semibold bg-blue-400/10 px-3 py-1 rounded-full">
          Accuracy: {tool.accuracy}%
        </span>
      </div>
      
      <h3 className="text-lg font-semibold text-white mb-4">{tool.title}</h3>
      
      <div className="space-y-3 mb-6">
        {tool.features.map((feature) => (
          <div key={feature} className="flex items-center text-sm text-gray-300">
            <Check className="w-4 h-4 text-green-400 mr-3 flex-shrink-0" />
            <span>{feature}</span>
          </div>
        ))}
      </div>
      
      <button 
        onClick={handleStartAnalysis}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center group-hover:shadow-lg group-hover:shadow-blue-500/25"
      >
        START ANALYSIS
        <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
      </button>
    </div>
  );
};

export default DetectionToolCard;