export type MediaType = 'image' | 'video' | 'audio';

export interface User {
  id: number;
  username: string;
  email?: string;
  role: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  token?: string;
  user?: User;
}

export interface AnalysisResult {
  id: string;
  type: MediaType;
  fileUrl: string;
  filename: string;
  confidenceScore: number;
  authenticity: 'AUTHENTIC' | 'MANIPULATED' | 'SUSPICIOUS' | 'UNKNOWN';
  technical_details?: {
    model_used: string;
    processing_time: number;
    detection_methods: string[];
  };
  timestamp: string | Date;
  metadata?: Record<string, any>;
}

export interface SystemStats {
  totalScans: number;
  authenticCount: number;
  manipulatedCount: number;
  suspiciousCount: number;
  averageConfidence: number;
  processingSpeed: number;
  systemHealth: number;
}
