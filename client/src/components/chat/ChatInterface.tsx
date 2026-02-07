import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader } from 'lucide-react';
import { useToast } from '../ui/use-toast';
import ChatMessage, { MessageStatus } from './ChatMessage';
import WelcomeMessage from './WelcomeMessage';
import { sendMessage, getChatHistory, Message as MessageType, ChatHistoryItem } from '../../services/chatService';
import { v4 as uuidv4 } from 'uuid';

interface Message extends MessageType {
  status?: MessageStatus;
  error?: string;
}

interface ChatInterfaceProps {
  initialPrompt?: string | null;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ initialPrompt }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Load conversation history on mount
  useEffect(() => {
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
        toast({
          variant: 'destructive',
          title: 'Error',
          description: 'Failed to load conversation history',
        });
      } finally {
        setIsLoadingHistory(false);
      }
    };

    loadConversation();
  }, [toast]);

  // Handle initial prompt from navigation
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

    // Validate message length (max 4000 characters)
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

      toast({
        title: 'Success',
        description: 'Message sent successfully',
      });
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

  const handlePromptSelect = useCallback((prompt: string) => {
    setInputValue(prompt);
    // Focus the input after selecting a prompt
    const input = document.querySelector('input[type="text"]') as HTMLInputElement;
    if (input) {
      input.focus();
    }
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-6 pb-4">
        {isLoadingHistory ? (
          <div className="flex justify-center items-center h-full">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#00a8ff]"></div>
          </div>
        ) : messages.length === 0 ? (
          <WelcomeMessage onPromptSelect={handlePromptSelect} />
        ) : (
          <div className="space-y-6">
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



      {/* Input Area */}
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
                className="w-full bg-[#0a0a0a] border border-[#333333] rounded-lg py-3 pl-4 pr-12 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#00a8ff]/50 focus:border-[#00a8ff]/30 transition-all duration-200"
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
    </div>
  );
};

export default ChatInterface;
