import { z } from 'zod';

// Common validation schemas
export const paginationSchema = z.object({
  page: z.preprocess(
    (val) => (typeof val === 'string' ? parseInt(val, 10) : val),
    z.number().int().positive().default(1)
  ),
  limit: z.preprocess(
    (val) => (typeof val === 'string' ? parseInt(val, 10) : val),
    z.number().int().positive().max(100).default(10)
  ),
});

// Auth schemas
export const loginSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email format'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
  }),
});

export const registerSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email format'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
    name: z.string().min(2, 'Name must be at least 2 characters'),
  }),
});

// Analysis schemas
export const analysisCreateSchema = z.object({
  body: z.object({
    type: z.enum(['sentiment', 'topic', 'summary']),
    text: z.string().min(1, 'Text cannot be empty'),
    options: z.record(z.unknown()).optional(),
  }),
});

export const analysisGetSchema = z.object({
  params: z.object({
    id: z.string().refine((val) => /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(val), {
      message: 'Invalid analysis ID format',
    }),
  }),
});

// Chat schemas
export const chatMessageSchema = z.object({
  body: z.object({
    message: z.string().min(1, 'Message cannot be empty'),
    conversationId: z.string().uuid('Invalid conversation ID format').optional(),
    context: z.record(z.unknown()).optional(),
  }),
});

export const chatConversationSchema = z.object({
  params: z.object({
    id: z.string().refine((val) => /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(val), {
      message: 'Invalid conversation ID format',
    }),
  }),
  query: z.object({
    limit: z.preprocess(
      (val) => (typeof val === 'string' ? parseInt(val, 10) : val),
      z.number().int().positive().max(100).default(20)
    ).optional(),
    before: z.string().datetime('Invalid date format').optional(),
  }),
});

// Export all schemas
export const validationSchemas = {
  auth: {
    login: loginSchema,
    register: registerSchema,
  },
  analysis: {
    create: analysisCreateSchema,
    get: analysisGetSchema,
  },
  chat: {
    message: chatMessageSchema,
    conversation: chatConversationSchema,
  },
  common: {
    pagination: paginationSchema,
  },
} as const;

// Type exports
export type LoginInput = z.infer<typeof loginSchema>;
export type RegisterInput = z.infer<typeof registerSchema>;
export type AnalysisCreateInput = z.infer<typeof analysisCreateSchema>;
export type ChatMessageInput = z.infer<typeof chatMessageSchema>;
