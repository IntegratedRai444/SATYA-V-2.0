import { Router, type Request, type Response } from 'express';
import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import { supabaseAuth } from '../middleware/supabase-auth';

// Extend the Express Request type to include user
interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
    email_verified: boolean;
    user_metadata?: Record<string, any>;
    [key: string]: any; // Allow additional properties
  };
}
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

// Check if OpenAI is properly configured
const isChatEnabled = process.env.ENABLE_CHAT === 'true' && process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'sk-placeholder-key-replace-with-real-key' && process.env.OPENAI_API_KEY.startsWith('sk-');

// In-memory conversation storage (use database in production)
const conversations = new Map<string, ChatCompletionMessageParam[]>();

// POST /api/chat/message - Send message to AI assistant
router.post('/message', chatRateLimit, supabaseAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Check if chat is enabled
    if (!isChatEnabled) {
      return res.status(503).json({
        success: false,
        error: 'Chat service is currently unavailable. Please configure OPENAI_API_KEY to enable chat functionality.',
        code: 'CHAT_DISABLED'
      });
    }

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
          content: `You are "Satya Sentinel", the assistant inside SatyaAI.

GOAL
Help users detect and understand manipulated or synthetic media (deepfakes) and guide them through SatyaAI features.

TONE & STYLE
- Friendly, calm, confident.
- Concise and clear. No long lectures.
- Use emojis occasionally (max 1–2 per message).
- Never brag about being "expert" or "advanced". Be humble.
- If uncertain, say so. Do not overclaim.

GREETING RULE
- Greet only ONCE at the start of a new conversation, OR when the user says "hi/hello".
- Greeting text must be exactly:
  "Hi there 👋 I'm Satya Sentinel. How can I help you today?"

CORE BEHAVIOR
- Always prioritize the user's intent: verification, explanation, safety steps, troubleshooting, or navigation.
- Explain results in simple language.
- If user asks "is it real or fake?", provide:
  - a likelihood (not 100% certainty)
  - evidence/signals
  - next steps for verification
  - a short note about probabilistic detection

LIMITATIONS
- Satya Sentinel cannot guarantee truth. Detection is probabilistic.
- Never claim legal certainty or court-proof confirmation.
- If user needs legal certainty: recommend professional forensic verification.

PRIVACY
- Treat user files/media as private.
- Do not ask for sensitive personal data unless required for support.

SAFETY RULES
Refuse requests involving:
- creating or assisting deepfakes to harm/harass or exploit someone
- fraud, identity impersonation, blackmail
- bypassing security systems or hacking
When refusing: be polite, brief, and suggest safe alternatives.

RESPONSE STRUCTURE
For analysis / real-vs-fake questions, respond in this structure:
1) Quick Answer (1 line)
2) Key Signals (bullets)
3) Next Steps (bullets)
4) Note (1 short line)

You are Satya Sentinel.`
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
router.get('/history', supabaseAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Check if chat is enabled
    if (!isChatEnabled) {
      return res.status(503).json({
        success: false,
        error: 'Chat service is currently unavailable. Please configure OPENAI_API_KEY to enable chat functionality.',
        code: 'CHAT_DISABLED'
      });
    }

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
router.get('/conversation/:id', supabaseAuth, async (req: AuthenticatedRequest, res: Response) => {
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
router.delete('/history/:id', supabaseAuth, async (req: AuthenticatedRequest, res: Response) => {
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
router.post('/suggestions', supabaseAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { message, conversationContext } = req.body;
    
    // Generate contextual suggestions based on the message
    const suggestions = [
      "Can you explain this analysis result in more detail?",
      "What confidence level should I be concerned about?",
      "How does this compare to other similar analyses?",
      "What steps should I take based on this result?"
    ];
    
    res.json({
      success: true,
      data: suggestions
    });
  } catch (error) {
    logger.error('Chat suggestions error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate suggestions'
    });
  }
});

export { router as chatRouter };
