import React from 'react';
import { Shield, Sparkles } from 'lucide-react';

interface WelcomeMessageProps {
  onPromptSelect: (prompt: string) => void;
}

const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ onPromptSelect }) => {
  const suggestions = [
    "How does deepfake detection work?",
    "What are signs of manipulated media?",
    "How accurate is AI detection?"
  ];

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a]">
      {/* Welcome Content */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-8">
        <div className="text-center max-w-md">
          {/* Icon */}
          <div className="w-12 h-12 rounded-lg bg-[#00a8ff]/10 border border-[#00a8ff]/20 flex items-center justify-center mx-auto mb-4">
            <Shield className="w-6 h-6 text-[#00a8ff]" strokeWidth={2} />
          </div>
          
          {/* Title */}
          <h2 className="text-[20px] font-semibold text-white mb-2">
            How can I help you today?
          </h2>
          
          {/* Description */}
          <p className="text-[13px] text-gray-400 mb-8 leading-relaxed">
            Ask me about deepfake detection, media analysis, or how to use SatyaAI features.
          </p>

          {/* Suggestions */}
          <div className="space-y-2">
            <p className="text-[11px] text-gray-500 mb-3 flex items-center justify-center gap-1">
              <Sparkles className="w-3 h-3 text-[#00a8ff]" />
              Suggested questions
            </p>
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => onPromptSelect(suggestion)}
                className="w-full text-left px-4 py-3 bg-[#0f1419] border border-[#333333] rounded-lg text-[13px] text-gray-300 hover:border-[#00a8ff]/40 hover:bg-[#00a8ff]/5 transition-all duration-200"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WelcomeMessage;
