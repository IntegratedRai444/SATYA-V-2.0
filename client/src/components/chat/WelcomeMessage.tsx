import React from 'react';
import { Bot } from 'lucide-react';

interface WelcomeMessageProps {
  onPromptSelect: (prompt: string) => void;
}

const suggestedPrompts = [
  'How does deepfake detection work?',
  'What are the signs of a manipulated video?',
  'How accurate is AI deepfake detection?',
  'What should I do if I suspect a deepfake?',
  'Explain confidence scores in analysis',
  'How can I protect myself from deepfakes?'
];

export const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ onPromptSelect }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div className="w-16 h-16 rounded-full bg-purple-900/50 flex items-center justify-center mb-6">
        <Bot className="text-3xl text-purple-400" />
      </div>
      <h1 className="text-3xl font-bold text-white mb-2">I'm Satya Sentinel 🛡️</h1>
      <p className="text-gray-400 mb-8 max-w-md">
        Your AI assistant for deepfake detection and media analysis. Ask me anything about identifying manipulated content or using SatyaAI features.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl mb-12">
        {suggestedPrompts.map((prompt, index) => (
          <button
            key={index}
            onClick={() => onPromptSelect(prompt)}
            className="p-4 text-left rounded-lg border border-gray-700 hover:border-purple-500 hover:bg-gray-800/50 transition-colors text-gray-200"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeMessage;
