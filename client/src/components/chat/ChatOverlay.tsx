import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Minimize2, Maximize2, Send, Loader } from 'lucide-react';
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
        bg-gray-900 border border-gray-700 rounded-lg shadow-2xl overflow-hidden
        transition-all duration-300 ease-in-out
        ${isMinimized ? 'w-80 h-14' : 'w-96 h-[600px] max-h-[80vh]'}
      `}>
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-900/20 to-cyan-900/20 border-b border-emerald-500/20 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
              <MessageCircle className="w-3 h-3 text-white" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-white">Satya Sentinel</h3>
              {!isMinimized && (
                <p className="text-xs text-gray-400">AI Assistant</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={toggleMinimize}
              className="p-1 hover:bg-gray-800 rounded transition-colors"
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
              className="p-1 hover:bg-gray-800 rounded transition-colors"
              title="Close"
            >
              <X className="w-4 h-4 text-gray-400" />
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
                  <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-purple-500"></div>
                </div>
              ) : messages.length === 0 ? (
                <WelcomeMessage onPromptSelect={handlePromptSelect} />
              ) : (
                <div className="space-y-4">
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
            <div className="p-3 border-t border-gray-800">
              <form onSubmit={handleSendMessage} className="relative">
                <div className="flex items-end space-x-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder="Message Satya Sentinel..."
                      maxLength={4000}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 pl-3 pr-10 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
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
                      className={`absolute right-1 bottom-1 p-1.5 rounded-md flex items-center justify-center ${
                        inputValue.trim() && !isSending
                          ? 'text-white bg-purple-600 hover:bg-purple-700'
                          : 'text-gray-400 bg-gray-700'
                      }`}
                    >
                      {isSending ? (
                        <Loader className="w-3 h-3 animate-spin" />
                      ) : (
                        <Send className="w-3 h-3" />
                      )}
                    </button>
                  </div>
                </div>
              </form>
              <div className="flex items-center justify-between mt-1">
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
