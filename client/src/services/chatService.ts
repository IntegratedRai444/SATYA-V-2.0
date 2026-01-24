import api from '../lib/api';
import { v4 as uuidv4 } from 'uuid';
import logger from '../lib/logger';

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  error?: string;
}

export interface ChatResponse {
  response: string;
  conversationId: string;
  sources?: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
  suggestions?: string[];
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

    const response = await api.post<ChatResponse>(
      '/api/v2/chat/message',
      formData
    );

    if (!response.response) {
      throw new Error('Failed to send message');
    }

    return {
      response: response.response || '',
      conversationId: (response as { conversationId?: string })?.conversationId || conversationId || uuidv4(),
      messageId: uuidv4(),
      sources: response.sources,
      suggestions: response.suggestions,
    };
  } catch (error) {
    logger.error('Error sending message', error as Error);
    throw error;
  }
};

/**
 * Get chat history for the current user
 */
export const getChatHistory = async (): Promise<ChatHistoryItem[]> => {
  try {
    const response = await api.get<{
      data: ChatHistoryItem[];
    }>('/api/v2/chat/history');

    return response.data;
  } catch (error) {
    logger.error('Error fetching chat history', error as Error);
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
    const response = await api.get<{
      data: Message[];
    }>(`/chat/conversation/${conversationId}`);

    return response.data.map((msg) => ({
      ...msg,
      timestamp: new Date(msg.timestamp),
    }));
  } catch (error) {
    logger.error('Error fetching conversation', error as Error);
    return [];
  }
};

/**
 * Delete a conversation
 */
export const deleteConversation = async (
  conversationId: string
): Promise<{ success: boolean; message: string }> => {
  try {
    const response = await api.delete<{
      message: string;
    }>(`/chat/conversation/${conversationId}`);

    return { success: true, message: response.message };
  } catch (error: unknown) {
    logger.error('Error deleting conversation', error as Error);
    return { success: false, message: (error as Error).message };
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
    const response = await api.post<{
      data: string[];
    }>('/api/v2/chat/suggestions', {
      message,
      conversationContext,
    });

    return response.data;
  } catch (error) {
    logger.error('Error getting suggested responses', error as Error);
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
