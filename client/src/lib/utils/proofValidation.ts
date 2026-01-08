import { toast } from '@/components/ui/use-toast';

export interface ProofOfAnalysis {
  model_name: string;
  model_version: string;
  modality: string;
  timestamp: string;
  inference_duration: number;
  frames_analyzed: number;
  signature: string;
  metadata: {
    request_id: string;
    user_id: string;
    analysis_type: string;
    content_size: number;
  };
}

export function validateProof(proof: any): { isValid: boolean; error?: string } {
  try {
    // Check if proof exists
    if (!proof) {
      return { isValid: false, error: 'Proof of analysis is missing' };
    }

    // Check required fields
    const requiredFields: (keyof ProofOfAnalysis)[] = [
      'model_name',
      'model_version',
      'modality',
      'timestamp',
      'inference_duration',
      'frames_analyzed',
      'signature',
      'metadata',
    ];

    for (const field of requiredFields) {
      if (proof[field] === undefined || proof[field] === null) {
        return { isValid: false, error: `Missing required proof field: ${field}` };
      }
    }

    // Validate inference duration
    if (proof.inference_duration <= 0) {
      return { isValid: false, error: 'Invalid inference duration in proof' };
    }

    // Validate timestamp is not in the future
    const proofTime = new Date(proof.timestamp).getTime();
    if (isNaN(proofTime) || proofTime > Date.now()) {
      return { isValid: false, error: 'Invalid timestamp in proof' };
    }

    // Verify proof is not too old (24 hours max)
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    if (Date.now() - proofTime > maxAge) {
      return { isValid: false, error: 'Proof has expired' };
    }

    // Verify signature (in a real app, this would verify against a public key)
    // For now, we'll just check it exists and has the right format
    if (typeof proof.signature !== 'string' || !proof.signature.startsWith('sig_')) {
      return { isValid: false, error: 'Invalid proof signature' };
    }

    return { isValid: true };
  } catch (error) {
    console.error('Error validating proof:', error);
    return { isValid: false, error: 'Failed to validate proof' };
  }
}

export function handleProofError(error: string): void {
  console.error('Proof validation failed:', error);
  toast({
    title: 'Analysis Error',
    description: 'Could not verify the authenticity of the analysis results. Please try again.',
    variant: 'destructive',
  });
}
