import { Router, type Request, type Response } from 'express';
import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import { supabaseAuth } from '../middleware/supabase-auth';
import { supabase } from '../config/supabase';
import { auditLogger } from '../middleware/audit-logger';

// Extend the Express Request type to include user
interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
    email_verified: boolean;
    user_metadata?: Record<string, unknown>;
    [key: string]: unknown; // Allow additional properties
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

// Database conversation storage
interface DatabaseConversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

// Helper functions for database operations
const createConversation = async (userId: string, title: string): Promise<string> => {
  const { data, error } = await supabase
    .from('chat_conversations')
    .insert({
      user_id: userId,
      title: title || 'New Chat'
    })
    .select('id')
    .single();
  
  if (error) throw error;
  return data.id;
};

const getConversationMessages = async (conversationId: string): Promise<ChatCompletionMessageParam[]> => {
  const { data, error } = await supabase
    .from('chat_messages')
    .select('role, content, created_at')
    .eq('conversation_id', conversationId)
    .order('created_at', { ascending: true });
  
  if (error) throw error;
  
  return data.map(msg => ({
    role: msg.role as 'user' | 'assistant' | 'system',
    content: msg.content
  }));
};

const saveMessage = async (conversationId: string, role: 'user' | 'assistant', content: string): Promise<void> => {
  const { error } = await supabase
    .from('chat_messages')
    .insert({
      conversation_id: conversationId,
      role,
      content
    });
  
  if (error) throw error;
  
  // Update conversation timestamp
  await supabase
    .from('chat_conversations')
    .update({ updated_at: new Date().toISOString() })
    .eq('id', conversationId);
};

const getUserConversations = async (userId: string): Promise<DatabaseConversation[]> => {
  const { data, error } = await supabase
    .from('chat_conversations')
    .select('id, title, created_at, updated_at')
    .eq('user_id', userId)
    .order('updated_at', { ascending: false });
  
  if (error) throw error;
  return data as DatabaseConversation[];
};

// POST /api/chat/message - Send message to AI assistant
router.post('/message', chatRateLimit, supabaseAuth, auditLogger('chat_message_send', 'chat_message'), async (req: AuthenticatedRequest, res: Response) => {
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

    // Get or create conversation
    let convId = conversationId;
    let history: ChatCompletionMessageParam[] = [];
    
    if (convId) {
      // Load existing conversation
      try {
        history = await getConversationMessages(convId);
      } catch (error) {
        logger.error('Failed to load conversation:', error);
        // If conversation doesn't exist, create new one
        convId = null;
      }
    }
    
    if (!convId) {
      // Create new conversation
      convId = await createConversation(req.user!.id, 'New Chat');
      history = [];
    }

    // Save user message to database
    await saveMessage(convId, 'user', message);
    
    // Add user message to history for API call
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

    // Save assistant message to database
    await saveMessage(convId, 'assistant', assistantMessage);

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

  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to process message';
    console.error('Chat error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/chat/history - Get chat history
router.get('/history', supabaseAuth, auditLogger('sensitive_data_access', 'chat_conversations'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Check if chat is enabled
    if (!isChatEnabled) {
      return res.status(503).json({
        success: false,
        error: 'Chat service is currently unavailable. Please configure OPENAI_API_KEY to enable chat functionality.',
        code: 'CHAT_DISABLED'
      });
    }

    // Get user's conversations from database
    const conversations = await getUserConversations(req.user!.id);
    
    // Convert to history format with message previews
    const history = await Promise.all(
      conversations.map(async (conv) => {
        const messages = await getConversationMessages(conv.id);
        const firstMsg = messages[0];
        const lastMsg = messages[messages.length - 1];
        const firstContent = (typeof firstMsg?.content === 'string' ? firstMsg.content : '') || '';
        const lastContent = (typeof lastMsg?.content === 'string' ? lastMsg.content : '') || '';
        
        return {
          id: conv.id,
          title: firstContent.substring(0, 50) || 'New Conversation',
          timestamp: new Date(conv.updated_at),
          preview: lastContent.substring(0, 100) || ''
        };
      })
    );

    res.json({
      success: true,
      data: history
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch chat history';
    console.error('Chat history error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// GET /api/chat/conversation/:id - Get specific conversation
router.get('/conversation/:id', supabaseAuth, auditLogger('sensitive_data_access', 'chat_conversation'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Verify conversation belongs to user
    const { data: conversation, error: convError } = await supabase
      .from('chat_conversations')
      .select('id')
      .eq('id', id)
      .eq('user_id', userId)
      .single();
    
    if (convError || !conversation) {
      return res.status(404).json({
        success: false,
        error: 'Conversation not found'
      });
    }

    const history = await getConversationMessages(id);
    
    const messages = history.map((msg, index) => ({
      id: `${id}-${index}`,
      content: typeof msg.content === 'string' ? msg.content : '',
      isUser: msg.role === 'user',
      timestamp: new Date()
    }));

    res.json({
      success: true,
      data: messages
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch conversation';
    console.error('Get conversation error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// DELETE /api/chat/conversation/:id - Delete conversation
router.delete('/history/:id', supabaseAuth, auditLogger('admin_action', 'chat_conversation'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'User not authenticated'
      });
    }

    // Soft delete conversation
    const { error } = await supabase
      .from('chat_conversations')
      .update({ deleted_at: new Date().toISOString() })
      .eq('id', id)
      .eq('user_id', userId);
    
    if (error) {
      throw error;
    }

    // Soft delete messages
    await supabase
      .from('chat_messages')
      .update({ deleted_at: new Date().toISOString() })
      .eq('conversation_id', id);

    res.json({
      success: true,
      message: 'Conversation deleted'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to delete conversation';
    console.error('Delete conversation error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

// POST /api/chat/suggestions - Get suggested responses
router.post('/suggestions', supabaseAuth, auditLogger('sensitive_data_access', 'chat_suggestions'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Generate contextual suggestions based on common deepfake detection queries
    const suggestions = [
      'How does deepfake detection work?',
      'What file formats are supported?',
      'How accurate is the detection?',
      'What should I look for in fake media?',
      'Can you explain confidence scores?',
      'How do I protect myself from deepfakes?'
    ];

    res.json({
      success: true,
      data: suggestions
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to generate suggestions';
    console.error('Chat suggestions error:', error);
    res.status(500).json({
      success: false,
      error: errorMessage
    });
  }
});

export { router as chatRouter };
