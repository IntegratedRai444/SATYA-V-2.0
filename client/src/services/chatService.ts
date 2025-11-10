import apiClient from '../lib/api';
import { v4 as uuidv4 } from 'uuid';

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  error?: string;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  data?: {
    response: string;
    sources?: Array<{
      title: string;
      url: string;
      snippet: string;
    }>;
    suggestions?: string[];
  };
  error?: string;
}

export interface ChatHistoryItem {
  id: string;
  title: string;
  timestamp: Date;
  preview: string;
}

/**
 * Send a message to the chat API
 */
export const sendMessage = async (
  message: string,
  conversationId: string | null = null,
  files: File[] = [],
  options: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    includeSources?: boolean;
  } = {}
): Promise<{
  response: string;
  conversationId: string;
  messageId: string;
  sources?: Array<{ title: string; url: string; snippet: string }>;
  suggestions?: string[];
}> => {
  try {
    const formData = new FormData();
    formData.append('message', message);
    
    if (conversationId) {
      formData.append('conversationId', conversationId);
    }

    // Append files if any
    files.forEach((file, index) => {
      formData.append(`file${index}`, file);
    });

    // Append options
    if (Object.keys(options).length > 0) {
      formData.append('options', JSON.stringify(options));
    }

    const response = await apiClient.client.post<ChatResponse>(
      '/api/chat/message',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to send message');
    }

    return {
      response: response.data.data?.response || '',
      conversationId: response.data.data?.conversationId || uuidv4(),
      messageId: uuidv4(),
      sources: response.data.data?.sources,
      suggestions: response.data.data?.suggestions,
    };
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

/**
 * Get chat history for the current user
 */
export const getChatHistory = async (): Promise<ChatHistoryItem[]> => {
  try {
    const response = await apiClient.client.get<{
      success: boolean;
      data: ChatHistoryItem[];
    }>('/api/chat/history');

    if (!response.data.success) {
      throw new Error('Failed to fetch chat history');
    }

    return response.data.data;
  } catch (error) {
    console.error('Error fetching chat history:', error);
    return [];
  }
};

/**
 * Get a specific conversation by ID
 */
export const getConversation = async (
  conversationId: string
): Promise<Message[]> => {
  try {
    const response = await apiClient.client.get<{
      success: boolean;
      data: Message[];
    }>(`/api/chat/conversation/${conversationId}`);

    if (!response.data.success) {
      throw new Error('Failed to fetch conversation');
    }

    return response.data.data.map((msg) => ({
      ...msg,
      timestamp: new Date(msg.timestamp),
    }));
  } catch (error) {
    console.error('Error fetching conversation:', error);
    return [];
  }
};

/**
 * Delete a conversation
 */
export const deleteConversation = async (
  conversationId: string
): Promise<boolean> => {
  try {
    const response = await apiClient.client.delete<{
      success: boolean;
      message: string;
    }>(`/api/chat/conversation/${conversationId}`);

    return response.data.success;
  } catch (error) {
    console.error('Error deleting conversation:', error);
    return false;
  }
};

/**
 * Get suggested responses for a message
 */
export const getSuggestedResponses = async (
  message: string,
  conversationContext: Message[] = []
): Promise<string[]> => {
  try {
    const response = await apiClient.client.post<{
      success: boolean;
      data: string[];
    }>('/api/chat/suggestions', {
      message,
      conversationContext,
    });

    if (!response.data.success) {
      return [];
    }

    return response.data.data;
  } catch (error) {
    console.error('Error getting suggested responses:', error);
    return [];
  }
};

export default {
  sendMessage,
  getChatHistory,
  getConversation,
  deleteConversation,
  getSuggestedResponses,
};
