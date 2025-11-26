import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import ChatInterface from '../components/chat/ChatInterface';
import { FiZap, FiMessageCircle } from 'react-icons/fi';

const AIAssistant: React.FC = () => {
  const location = useLocation();
  const [initialPrompt, setInitialPrompt] = useState<string | null>(null);

  useEffect(() => {
    // Check if there's a quick prompt from navigation state
    if (location.state && (location.state as any).quickPrompt) {
      setInitialPrompt((location.state as any).quickPrompt);
    }
  }, [location]);

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a]">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-900/20 to-cyan-900/20 border-b border-emerald-500/20 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
              <FiZap className="w-5 h-5 text-white" strokeWidth={2.5} />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#0a0a0a] animate-pulse"></div>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              SATYA AI Assistant
              <span className="text-xs font-normal text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">
                Online
              </span>
            </h1>
            <p className="text-sm text-gray-400">
              Your intelligent deepfake detection companion
            </p>
          </div>
        </div>
      </div>

      {/* Chat Interface */}
      <div className="flex-1 overflow-hidden">
        <ChatInterface initialPrompt={initialPrompt} />
      </div>

      {/* Footer Info */}
      <div className="bg-[#0f0f0f] border-t border-gray-800 px-6 py-3">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-2">
            <FiMessageCircle className="w-3.5 h-3.5" />
            <span>Powered by OpenAI GPT-4o-mini</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Model: gpt-4o-mini</span>
            <span className="text-emerald-400">● Connected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;
