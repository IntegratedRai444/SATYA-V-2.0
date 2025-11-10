import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FiSend, FiPaperclip, FiX, FiLoader } from 'react-icons/fi';
import { useToast } from '../ui/use-toast';
import FileUpload from '../ui/FileUpload';
import ChatMessage, { MessageStatus } from './ChatMessage';
import WelcomeMessage from './WelcomeMessage';
import { sendMessage, getChatHistory, Message as MessageType, ChatHistoryItem } from '../../services/chatService';
import { v4 as uuidv4 } from 'uuid';
import { useWebSocket } from '../../hooks/useWebSocket';

interface Message extends MessageType {
  status?: MessageStatus;
  error?: string;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isConnected: isWsConnected, sendMessage: sendWsMessage } = useWebSocket({ autoConnect: true });
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

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!inputValue.trim() && selectedFiles.length === 0) || isSending) return;

    const userMessage: Message = {
      id: `msg_${uuidv4()}`,
      content: inputValue,
      isUser: true,
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    
    const messageText = inputValue;
    const filesToSend = [...selectedFiles];
    setInputValue('');
    setSelectedFiles([]);
    setShowFileUpload(false);
    setIsSending(true);

    try {
      // Send via WebSocket if available
      if (isWsConnected) {
        sendWsMessage({
          type: 'new_message',
          data: {
            messageId: userMessage.id,
            content: messageText,
            conversationId,
            files: filesToSend.map(f => f.name)
          }
        });
      }

      const response = await sendMessage(
        messageText,
        conversationId,
        filesToSend,
        {
          model: 'gpt-4',
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

  const handleFilesSelected = useCallback((files: File[]) => {
    setSelectedFiles(files);
  }, []);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles(prev => {
      const newFiles = [...prev];
      newFiles.splice(index, 1);
      return newFiles;
    });
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-6 pb-4">
        {isLoadingHistory ? (
          <div className="flex justify-center items-center h-full">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
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

      {/* File Upload Area */}
      {showFileUpload && (
        <div className="px-6 pb-4">
          <div className="bg-gray-800/50 rounded-xl p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-medium text-white">Upload Files</h3>
              <button
                onClick={() => setShowFileUpload(false)}
                className="text-gray-400 hover:text-white"
              >
                <FiX />
              </button>
            </div>
            <FileUpload
              onFilesSelected={handleFilesSelected}
              multiple={true}
              maxSizeMB={50}
            />
          </div>
        </div>
      )}

      {/* Selected Files Preview */}
      {selectedFiles.length > 0 && !showFileUpload && (
        <div className="px-6 pb-2">
          <div className="flex flex-wrap gap-2">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center bg-gray-800/70 rounded-full px-3 py-1.5 text-sm"
              >
                <span className="text-purple-400 mr-2">
                  {file.name.length > 15 ? `${file.name.substring(0, 10)}...` : file.name}
                </span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-white"
                >
                  <FiX className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-t border-gray-800">
        <form onSubmit={handleSendMessage} className="relative">
          <div className="flex items-end space-x-2">
            <button
              type="button"
              onClick={() => setShowFileUpload(!showFileUpload)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg"
            >
              <FiPaperclip className="w-5 h-5" />
            </button>
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Message SATYA AI..."
                className="w-full bg-gray-800 border border-gray-700 rounded-xl py-3 pl-4 pr-12 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
              />
              <button
                type="submit"
                disabled={(!inputValue.trim() && selectedFiles.length === 0) || isSending}
                className={`absolute right-2 bottom-2 p-1.5 rounded-lg flex items-center justify-center ${
                  (inputValue.trim() || selectedFiles.length > 0) && !isSending
                    ? 'text-white bg-purple-600 hover:bg-purple-700'
                    : 'text-gray-600 bg-gray-700 cursor-not-allowed'
                }`}
              >
                {isSending ? (
                  <div className="flex items-center">
                    <FiLoader className="w-4 h-4 animate-spin mr-1" />
                    <span className="text-xs">Sending</span>
                  </div>
                ) : (
                  <FiSend className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </form>
        <p className="text-xs text-gray-500 text-center mt-2">
          SATYA AI can make mistakes. Consider checking important information.
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;
