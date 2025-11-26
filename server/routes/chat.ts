import { Router, type Request, type Response } from 'express';
import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import { optionalAuth, type AuthenticatedRequest } from '../middleware/auth';
import { logger } from '../config';
import rateLimit from 'express-rate-limit';

const router = Router();

// Rate limiter for chat endpoints
const chatRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 50, // 50 requests per window per IP
  message: 'Too many chat requests from this IP, please try again after 15 minutes',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
});

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
});

// In-memory conversation storage (use database in production)
const conversations = new Map<string, ChatCompletionMessageParam[]>();

// POST /api/chat/message - Send message to AI assistant
router.post('/message', chatRateLimit, optionalAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { message, conversationId, options } = req.body;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Message is required'
      });
    }

    // Validate message length (max 4000 characters)
    if (message.length > 4000) {
      return res.status(400).json({
        success: false,
        error: 'Message is too long. Maximum length is 4000 characters.'
      });
    }

    // Validate message is not empty after trimming
    if (message.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Message cannot be empty'
      });
    }

    // Get or create conversation history
    const convId = conversationId || `conv-${Date.now()}`;
    let history = conversations.get(convId) || [];

    // Add user message to history
    history.push({
      role: 'user',
      content: message
    });

    // Call OpenAI API
    const completion = await openai.chat.completions.create({
      model: options?.model || process.env.OPENAI_MODEL || 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful AI assistant for SatyaAI, a deepfake detection platform. Help users understand deepfake detection, analyze results, and use the platform effectively.'
        },
        ...history
      ],
      temperature: options?.temperature || 0.7,
      max_tokens: options?.maxTokens || 2000,
    });

    const assistantMessage = completion.choices[0]?.message?.content || 'Sorry, I could not generate a response.';

    // Add assistant response to history
    history.push({
      role: 'assistant',
      content: assistantMessage
    });

    // Store updated history
    conversations.set(convId, history);

    // Return response
    res.json({
      success: true,
      message: 'Message sent successfully',
      data: {
        response: assistantMessage,
        conversationId: convId,
        suggestions: [
          'How does deepfake detection work?',
          'What file formats are supported?',
          'How accurate is the detection?'
        ]
      }
    });

  } catch (error: any) {
    logger.error('Chat error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to process message'
    });
  }
});

// GET /api/chat/history - Get chat history
router.get('/history', optionalAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Convert conversations to history format
    const history = Array.from(conversations.entries()).map(([id, messages]) => {
      const firstMsg = messages[0];
      const lastMsg = messages[messages.length - 1];
      const firstContent = firstMsg && 'content' in firstMsg && typeof firstMsg.content === 'string' ? firstMsg.content : '';
      const lastContent = lastMsg && 'content' in lastMsg && typeof lastMsg.content === 'string' ? lastMsg.content : '';
      
      return {
        id,
        title: firstContent.substring(0, 50) || 'New Conversation',
        timestamp: new Date(),
        preview: lastContent.substring(0, 100) || ''
      };
    });

    res.json({
      success: true,
      data: history
    });
  } catch (error: any) {
    logger.error('Chat history error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch chat history'
    });
  }
});

// GET /api/chat/conversation/:id - Get specific conversation
router.get('/conversation/:id', optionalAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { id } = req.params;
    const history = conversations.get(id) || [];

    const messages = history.map((msg, index) => ({
      id: `${id}-${index}`,
      content: msg.content,
      isUser: msg.role === 'user',
      timestamp: new Date()
    }));

    res.json({
      success: true,
      data: messages
    });
  } catch (error: any) {
    logger.error('Get conversation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch conversation'
    });
  }
});

// DELETE /api/chat/conversation/:id - Delete conversation
router.delete('/conversation/:id', optionalAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { id } = req.params;
    conversations.delete(id);

    res.json({
      success: true,
      message: 'Conversation deleted'
    });
  } catch (error: any) {
    logger.error('Delete conversation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete conversation'
    });
  }
});

// POST /api/chat/suggestions - Get suggested responses
router.post('/suggestions', optionalAuth, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    res.json({
      success: true,
      data: [
        'Tell me about this analysis',
        'How can I improve detection accuracy?',
        'What should I look for in the results?'
      ]
    });
  } catch (error: any) {
    logger.error('Suggestions error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get suggestions'
    });
  }
});

export default router;
