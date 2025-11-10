import React from 'react';
import { RiRobot2Line } from 'react-icons/ri';

interface WelcomeMessageProps {
  onPromptSelect: (prompt: string) => void;
}

const suggestedPrompts = [
  'How can I improve my website?',
  'Create a marketing plan for my business',
  'Help me write a professional email',
  'Explain machine learning in simple terms'
];

export const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ onPromptSelect }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div className="w-16 h-16 rounded-full bg-purple-900/50 flex items-center justify-center mb-6">
        <RiRobot2Line className="text-3xl text-purple-400" />
      </div>
      <h1 className="text-3xl font-bold text-white mb-2">How can I help you today?</h1>
      <p className="text-gray-400 mb-8 max-w-md">
        I'm here to assist you with any questions or tasks you have. Feel free to ask me anything!
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
