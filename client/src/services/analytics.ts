import { webSocketService } from './websocket';

interface ScanStats {
  totalScans: number;
  successRate: number;
  failureRate: number;
  scansByType: {
    image: number;
    video: number;
    audio: number;
  };
  storageUsage: {
    used: number; // in MB
    total: number; // in MB
  };
  recentActivity: Array<{
    id: string;
    type: 'scan' | 'user' | 'system';
    action: string;
    timestamp: Date;
    metadata?: Record<string, any>;
  }>;
}

class AnalyticsService {
  private stats: ScanStats = {
    totalScans: 0,
    successRate: 0,
    failureRate: 0,
    scansByType: {
      image: 0,
      video: 0,
      audio: 0,
    },
    storageUsage: {
      used: 0,
      total: 1024, // 1GB default
    },
    recentActivity: [],
  };

  constructor() {
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    webSocketService.subscribe((data) => {
      if (data.type === 'scan_update') {
        this.handleScanUpdate(data.payload);
      } else if (data.type === 'storage_update') {
        this.handleStorageUpdate(data.payload);
      }
    });
  }

  private handleScanUpdate(scanData: any) {
    // Update scan statistics
    this.stats.totalScans++;
    
    // Update scan type count
    if (scanData.type in this.stats.scansByType) {
      this.stats.scansByType[scanData.type as keyof typeof this.stats.scansByType]++;
    }

    // Update success/failure rates
    const totalScans = this.stats.totalScans;
    const successfulScans = this.stats.recentActivity.filter(
      (activity) => activity.type === 'scan' && activity.metadata?.status === 'completed'
    ).length;
    
    this.stats.successRate = (successfulScans / totalScans) * 100;
    this.stats.failureRate = 100 - this.stats.successRate;

    // Add to recent activity
    this.stats.recentActivity.unshift({
      id: `scan-${Date.now()}`,
      type: 'scan',
      action: `New ${scanData.type} scan ${scanData.status}`,
      timestamp: new Date(),
      metadata: scanData,
    });

    // Keep only the 50 most recent activities
    if (this.stats.recentActivity.length > 50) {
      this.stats.recentActivity.pop();
    }
  }

  private handleStorageUpdate(storageData: any) {
    this.stats.storageUsage = {
      used: storageData.used,
      total: storageData.total,
    };
  }

  public getStats(): ScanStats {
    return this.stats;
  }

  public async fetchInitialData() {
    try {
      // In a real app, you would fetch this from your API
      const response = await fetch('/api/analytics');
      const data = await response.json();
      this.stats = { ...this.stats, ...data };
    } catch (error) {
      console.error('Failed to fetch initial analytics data:', error);
    }
  }
}

export const analyticsService = new AnalyticsService();
