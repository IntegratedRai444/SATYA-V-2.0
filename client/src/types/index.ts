// Global type definitions for SatyaAI

export interface AuthResponse {
    success: boolean;
    message: string;
    token?: string;
    user?: {
        id: string;
        username: string;
        email?: string;
        fullName?: string;
        role: string;
    };
    errors?: string[];
}

export interface User {
    id: string;
    username: string;
    email: string;
    fullName?: string;
    role: 'user' | 'admin' | 'moderator';
    createdAt?: string;
}

export interface ApiError {
    message: string;
    code?: string;
    status?: number;
}

export interface PaginatedResponse<T> {
    data: T[];
    total: number;
    page: number;
    pageSize: number;
    hasMore: boolean;
}

export type MediaType = 'image' | 'video' | 'audio' | 'multimodal';

export interface AnalysisResult {
    id: string;
    type: MediaType;
    result: 'authentic' | 'manipulated' | 'uncertain';
    confidence: number;
    timestamp: string;
    details?: Record<string, unknown>;
}
