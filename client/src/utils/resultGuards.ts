/**
 * Type guards and safe accessors for analysis results
 */

export interface SafeAnalysisResult {
  id: string;
  status: string;
  confidence: number;
  is_deepfake: boolean;
  model_name: string;
  model_version: string;
  summary: Record<string, unknown>;
  proof: Record<string, unknown> | null;
  file_name: string;
  created_at: string;
  error?: string;
}

export const safeAnalysisResult = (result: unknown): SafeAnalysisResult => {
  const data = result as Record<string, unknown>;
  
  return {
    id: (data.id as string) || 'unknown',
    status: (data.status as string) || 'unknown',
    confidence: typeof data.confidence === 'number' ? data.confidence : 0,
    is_deepfake: typeof data.is_deepfake === 'boolean' ? data.is_deepfake : false,
    model_name: (data.model_name as string) || 'SatyaAI',
    model_version: (data.model_version as string) || '1.0.0',
    summary: (data.summary as Record<string, unknown>) || {},
    proof: (data.proof as Record<string, unknown>) || null,
    file_name: (data.file_name as string) || 'Unknown file',
    created_at: (data.created_at as string) || new Date().toISOString(),
    error: (data.error as string) || undefined
  };
};

export const getConfidenceDisplay = (confidence: number): string => {
  if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) {
    return '0%';
  }
  return `${Math.round(confidence * 100)}%`;
};

export const getAuthenticityLabel = (isDeepfake: boolean): string => {
  return isDeepfake ? 'MANIPULATED MEDIA' : 'AUTHENTIC MEDIA';
};

export const getProofStatus = (proof: Record<string, unknown> | null): string => {
  if (!proof) return 'Proof unavailable';
  if (proof.signature) return 'Cryptographic proof verified';
  return 'Proof incomplete';
};

export const hasValidProof = (proof: Record<string, unknown> | null): boolean => {
  return !!(proof && proof.signature && proof.model_name && proof.timestamp);
};
