import React from 'react';
import { MessageCircle } from 'lucide-react';
import { useChat } from '@/contexts/ChatContext';

const FloatingChatButton: React.FC = () => {
  const { openChat } = useChat();

  return (
    <button
      onClick={() => openChat()}
      className="fixed bottom-6 right-6 z-40 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-700 hover:to-cyan-700 text-white rounded-full p-4 shadow-lg hover:shadow-xl transition-all duration-300 group"
      title="Open Satya Sentinel AI Assistant"
    >
      <MessageCircle className="w-6 h-6 group-hover:scale-110 transition-transform" />
      <span className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-white animate-pulse"></span>
    </button>
  );
};

export default FloatingChatButton;
