import React from 'react';
import { User, Shield, Loader, AlertCircle, Check } from 'lucide-react';

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
  sending: <Loader className="w-3 h-3 animate-spin text-[#00a8ff]" />,
  sent: <Check className="w-3 h-3 text-[#00a8ff]" />,
  error: <AlertCircle className="w-3 h-3 text-red-400" />,
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
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 ${className}`}
      data-is-user={isUser}
      data-status={status}
    >
      <div className={`flex max-w-2xl ${isUser ? 'flex-row-reverse' : ''} items-end gap-2`}>
        {/* Avatar */}
        <div 
          className={`w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 ${
            isUser ? 'bg-blue-500/10 border border-blue-500/20 ml-2' : 'bg-[#00a8ff]/10 border border-[#00a8ff]/20 mr-2'
          }`}
        >
          {isUser ? 
            <User className="w-3 h-3 text-blue-400" /> : 
            <Shield className="w-3 h-3 text-[#00a8ff]" />
          }
        </div>
        
        {/* Message Bubble */}
        <div 
          className={`px-4 py-2 rounded-lg ${
            isUser 
              ? 'bg-blue-500/10 border border-blue-500/20' 
              : 'bg-[#0f1419] border border-[#333333]'
          }`}
        >
          <p className="text-[13px] text-gray-100 leading-relaxed">{content}</p>
          <div className="flex items-center justify-between mt-1">
            <p className="text-[10px] text-gray-500">
              {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </p>
            {statusIcon && (
              <span className="ml-2">
                {statusIcon}
              </span>
            )}
          </div>
          {error && (
            <p className="text-[10px] text-red-400 mt-1">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
