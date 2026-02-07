import React from 'react';
import { Shield } from 'lucide-react';
import { useChat } from '@/contexts/ChatContext';

const FloatingChatButton: React.FC = () => {
  const { openChat } = useChat();

  return (
    <button
      onClick={() => openChat()}
      className="fixed bottom-6 right-6 z-40 bg-[#00a8ff] hover:bg-[#0088cc] text-white rounded-lg p-4 shadow-lg hover:shadow-xl transition-all duration-300 group"
      title="Open Satya Sentinel AI Assistant"
    >
      <div className="relative">
        <Shield className="w-6 h-6 group-hover:scale-110 transition-transform" />
        <div className="absolute -top-1 -right-1 w-3 h-3 bg-white rounded-full border-2 border-[#00a8ff] animate-pulse"></div>
      </div>
    </button>
  );
};

export default FloatingChatButton;
