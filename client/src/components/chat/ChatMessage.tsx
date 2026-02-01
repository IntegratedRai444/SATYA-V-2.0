import React from 'react';
import { User, Cpu, Loader, AlertCircle, Check } from 'lucide-react';

export type MessageStatus = 'sending' | 'sent' | 'error';

export interface ChatMessageProps {
  content: string;
  isUser: boolean;
  timestamp?: Date;
  status?: MessageStatus;
  error?: string;
  className?: string;
}

const statusIcons = {
  sending: <Loader className="w-3 h-3 animate-spin" />,
  sent: <Check className="w-3 h-3 text-green-500" />,
  error: <AlertCircle className="w-3 h-3 text-red-500" />,
};

const ChatMessage: React.FC<ChatMessageProps> = ({
  content,
  isUser,
  timestamp = new Date(),
  status,
  error,
  className = '',
}) => {
  const statusIcon = status ? statusIcons[status] : null;

  return (
    <div 
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 ${className}`}
      data-is-user={isUser}
      data-status={status}
    >
      <div className={`flex max-w-3xl ${isUser ? 'flex-row-reverse' : ''}`}>
        <div 
          className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
            isUser ? 'bg-purple-600 ml-3' : 'bg-gray-700 mr-3'
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
        </div>
        <div 
          className={`px-4 py-3 rounded-2xl relative ${
            isUser ? 'bg-purple-900/50 rounded-tr-none' : 'bg-gray-800 rounded-tl-none'
          }`}
        >
          <p className="text-gray-100">{content}</p>
          <div className="flex items-center justify-between mt-1">
            <p className="text-xs text-gray-400">
              {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </p>
            {statusIcon && (
              <span className="ml-2">
                {statusIcon}
              </span>
            )}
          </div>
          {error && (
            <p className="text-xs text-red-400 mt-1">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
