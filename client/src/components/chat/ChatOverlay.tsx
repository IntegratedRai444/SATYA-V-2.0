import React, { useState, useRef, useEffect } from 'react';
import { Shield, X, Minimize2, Maximize2, Send, Loader } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import ChatMessage, { MessageStatus } from './ChatMessage';
import WelcomeMessage from './WelcomeMessage';
import { sendMessage, getChatHistory, Message as MessageType, ChatHistoryItem } from '@/services/chatService';
import { v4 as uuidv4 } from 'uuid';

interface Message extends MessageType {
  status?: MessageStatus;
  error?: string;
}

interface ChatOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  initialPrompt?: string | null;
}

const ChatOverlay: React.FC<ChatOverlayProps> = ({ isOpen, onClose, initialPrompt }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isMinimized, setIsMinimized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Load conversation history on mount
  useEffect(() => {
    if (isOpen) {
      const loadConversation = async () => {
        try {
          const history = await getChatHistory();
          if (history && history.length > 0) {
            const recentChat = history[0];
            setConversationId(recentChat.id);
            
            const mappedMessages: Message[] = history.map((chat: ChatHistoryItem) => ({
              id: chat.id,
              content: chat.preview || 'No content',
              isUser: false,
              timestamp: chat.timestamp instanceof Date ? chat.timestamp : new Date(chat.timestamp)
            }));
            
            setMessages(mappedMessages);
          }
        } catch (error) {
          console.error('Failed to load conversation:', error);
        } finally {
          setIsLoadingHistory(false);
        }
      };

      loadConversation();
    }
  }, [isOpen]);

  // Handle initial prompt
  useEffect(() => {
    if (initialPrompt && !isSending && messages.length === 0) {
      setInputValue(initialPrompt);
    }
  }, [initialPrompt, isSending, messages.length]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isSending) return;

    // Validate message length
    if (inputValue.length > 4000) {
      toast({
        variant: 'destructive',
        title: 'Message too long',
        description: 'Please keep your message under 4000 characters.',
      });
      return;
    }

    const userMessage: Message = {
      id: `msg_${uuidv4()}`,
      content: inputValue,
      isUser: true,
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    
    const messageText = inputValue;
    setInputValue('');
    setIsSending(true);

    try {
      const response = await sendMessage(
        messageText,
        conversationId,
        [],
        {
          model: 'gpt-4o-mini',
          temperature: 0.7,
          includeSources: true
        }
      );

      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'sent' as const } 
            : msg
        )
      );

      const botMessage: Message = {
        id: `msg_${uuidv4()}`,
        content: response.response,
        isUser: false,
        timestamp: new Date(),
        status: 'sent'
      };

      setMessages(prev => [...prev, botMessage]);
      
      if (!conversationId) {
        setConversationId(response.conversationId);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { 
                ...msg, 
                status: 'error', 
                error: errorMessage
              } 
            : msg
        )
      );
      
      toast({
        variant: 'destructive',
        title: 'Error',
        description: errorMessage,
      });
    } finally {
      setIsSending(false);
    }
  };

  const handlePromptSelect = (prompt: string) => {
    setInputValue(prompt);
  };

  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col">
      {/* Chat Window */}
      <div className={`
        bg-[#0a0a0a] border border-[#333333] rounded-lg shadow-lg overflow-hidden
        transition-all duration-300 ease-in-out
        ${isMinimized ? 'w-80 h-14' : 'w-96 h-[600px] max-h-[80vh]'}
      `}>
        {/* Header */}
        <div className="bg-[#0f1419] border-b border-[#333333] px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#00a8ff]/10 border border-[#00a8ff]/20 flex items-center justify-center">
              <Shield className="w-4 h-4 text-[#00a8ff]" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <h3 className="text-[15px] font-semibold text-white">Satya Sentinel</h3>
              {!isMinimized && (
                <p className="text-[11px] text-gray-500">AI Assistant</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={toggleMinimize}
              className="p-1.5 hover:bg-[#333333] rounded-md transition-colors"
              title={isMinimized ? "Maximize" : "Minimize"}
            >
              {isMinimized ? (
                <Maximize2 className="w-4 h-4 text-gray-400" />
              ) : (
                <Minimize2 className="w-4 h-4 text-gray-400" />
              )}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-red-500/10 rounded-md transition-colors"
              title="Close"
            >
              <X className="w-4 h-4 text-gray-400 hover:text-red-400" />
            </button>
          </div>
        </div>

        {/* Chat Content (hidden when minimized) */}
        {!isMinimized && (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 pb-2">
              {isLoadingHistory ? (
                <div className="flex justify-center items-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-[#00a8ff]"></div>
                </div>
              ) : messages.length === 0 ? (
                <WelcomeMessage onPromptSelect={handlePromptSelect} />
              ) : (
                <div className="space-y-3">
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      content={message.content}
                      isUser={message.isUser}
                      timestamp={message.timestamp}
                      status={message.status}
                      error={message.error}
                    />
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-[#333333] bg-[#0f1419]">
              <form onSubmit={handleSendMessage} className="relative">
                <div className="flex items-end space-x-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder="Ask Satya Sentinel about deepfakes…"
                      maxLength={4000}
                      className="w-full bg-[#0a0a0a] border border-[#333333] rounded-lg py-3 pl-4 pr-12 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#00a8ff]/50 focus:border-[#00a8ff]/30 transition-all duration-200"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage(e);
                        }
                      }}
                    />
                    <button
                      type="submit"
                      disabled={!inputValue.trim() || isSending}
                      className={`absolute right-2 bottom-2 p-2 rounded-lg flex items-center justify-center transition-all duration-200 ${
                        inputValue.trim() && !isSending
                          ? 'text-white bg-[#00a8ff] hover:bg-[#0088cc]'
                          : 'text-gray-400 bg-[#333333]'
                      }`}
                    >
                      {isSending ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              </form>
              <div className="flex items-center justify-between mt-2">
                <p className="text-xs text-gray-500">
                  Satya Sentinel can make mistakes. Consider checking important information.
                </p>
                <p className="text-xs text-gray-500">
                  {inputValue.length}/4000
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ChatOverlay;
