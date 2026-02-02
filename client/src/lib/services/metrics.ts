import { v4 as uuidv4 } from 'uuid';
import api from '@/lib/api';

// Types
export type MetricType = 'analysis' | 'error' | 'performance' | 'auth' | 'api' | 'health';

export interface MetricPayload {
  type: MetricType;
  name: string;
  value: number;
  metadata?: Record<string, unknown>;
  timestamp?: number;
  sessionId?: string;
  userId?: string | null;
}

class MetricsService {
  private sessionId: string;
  private enabled: boolean;
  private queue: MetricPayload[] = [];
  private isProcessing = false;
  private readonly BATCH_SIZE = 10;
  private readonly FLUSH_INTERVAL = 30000; // 30 seconds
  private flushTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.sessionId = this.getOrCreateSessionId();
    this.enabled = import.meta.env.PROD;
    this.initialize();
  }

  private getOrCreateSessionId(): string {
    let sessionId = localStorage.getItem('metrics_session_id');
    if (!sessionId) {
      sessionId = uuidv4();
      localStorage.setItem('metrics_session_id', sessionId);
    }
    return sessionId;
  }

  private initialize() {
    if (typeof window === 'undefined') return;
    
    // Flush metrics periodically
    this.flushTimer = setInterval(() => this.flush(), this.FLUSH_INTERVAL);
    
    // Flush metrics when page is hidden or unloaded
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    };
    
    const handleBeforeUnload = () => {
      this.flushSync();
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // Cleanup
    return () => {
      if (this.flushTimer) clearInterval(this.flushTimer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }

  track(payload: Omit<MetricPayload, 'sessionId' | 'timestamp'>) {
    if (!this.enabled) return;
    
    const metric: MetricPayload = {
      ...payload,
      sessionId: this.sessionId,
      timestamp: Date.now(),
      userId: this.getUserId(),
    };
    
    this.queue.push(metric);
    
    // Flush if queue reaches batch size
    if (this.queue.length >= this.BATCH_SIZE) {
      this.flush();
    }
  }

  private getUserId(): string | null {
    // Implement your own logic to get current user ID
    return localStorage.getItem('user_id');
  }

  private async flush() {
    if (this.isProcessing || this.queue.length === 0) return;
    
    this.isProcessing = true;
    const batch = this.queue.splice(0, this.BATCH_SIZE);
    
    try {
      await api.post('/api/metrics', { metrics: batch });
    } catch (error: unknown) {
      console.error('Failed to send metrics:', error);
      // Requeue failed metrics (except in case of client errors)
      const status = (error as { response?: { status?: number } })?.response?.status;
      if (status && status >= 500) {
        this.queue.unshift(...batch);
      }
    } finally {
      this.isProcessing = false;
      
      // If there are more metrics in the queue, schedule next flush
      if (this.queue.length > 0) {
        setTimeout(() => this.flush(), 0);
      }
    }
  }

  // Synchronous flush for critical events
  flushSync() {
    if (this.queue.length === 0) return;
    
    // Use sendBeacon for reliable delivery during page unload
    const blob = new Blob(
      [JSON.stringify({ metrics: this.queue })],
      { type: 'application/json' }
    );
    
    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/metrics', blob);
      this.queue = [];
    } else {
      // Fallback to regular fetch
      fetch('/api/metrics', {
        method: 'POST',
        body: blob,
        keepalive: true,
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
  }
}

// Singleton instance
export const metrics = new MetricsService();

// Helper functions
export const trackAnalysis = (
  name: string, 
  duration: number, 
  type: 'image' | 'video' | 'audio' | 'multimodal',
  success: boolean,
  metadata: Record<string, unknown> = {}
) => {
  metrics.track({
    type: 'analysis',
    name: `analysis.${type}.${name}`,
    value: duration,
    metadata: {
      ...metadata,
      analysisType: type,
      success,
    },
  });};

export const trackError = (
  error: Error,
  context: string,
  metadata: Record<string, unknown> = {}
) => {
  metrics.track({
    type: 'error',
    name: `error.${context}`,
    value: 1,
    metadata: {
      ...metadata,
      error: error.message,
      stack: error.stack,
      context,
    },
  });
};

export const trackPerformance = (
  name: string,
  duration: number,
  metadata: Record<string, unknown> = {}
) => {
  metrics.track({
    type: 'performance',
    name: `perf.${name}`,
    value: duration,
    metadata,
  });
};
