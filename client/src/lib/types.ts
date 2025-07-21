export interface ScanResult {
  id: string;
  filename: string;
  type: 'image' | 'video' | 'audio';
  result: 'authentic' | 'deepfake';
  confidenceScore: number;
  timestamp: string;
  detectionDetails?: DetectionDetail[];
  metadata?: {
    resolution?: string;
    duration?: string;
    size?: string;
  };
}

export interface DetectionDetail {
  name: string;
  category?: 'face' | 'audio' | 'frame' | 'general';
  confidence: number;
  description: string;
}
