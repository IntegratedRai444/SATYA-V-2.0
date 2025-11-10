export interface DashboardStats {
  analyzedMedia: {
    count: number;
    growth: string;
    growthType?: 'positive' | 'negative';
  };
  detectedDeepfakes: {
    count: number;
    growth: string;
    growthType?: 'positive' | 'negative';
  };
  avgDetectionTime: {
    time: string;
    improvement: string;
    improvementType?: 'positive' | 'negative';
  };
  detectionAccuracy: {
    percentage: number;
    improvement: string;
    improvementType?: 'positive' | 'negative';
  };
  dailyActivity: Array<{
    date: string;
    analyses: number;
    detections: number;
  }>;
}

export interface Detection {
  id: string;
  type: 'image' | 'video' | 'audio';
  status: 'completed' | 'processing' | 'failed';
  timestamp: string;
  confidence: number;
  // Add other detection properties as needed
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}
