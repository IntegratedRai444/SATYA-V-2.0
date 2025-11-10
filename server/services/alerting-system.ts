import { EventEmitter } from 'events';
import { logger } from '../config/logger';

export interface Alert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  component: string;
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
  metadata?: Record<string, any>;
}

export interface AlertRule {
  id: string;
  name: string;
  condition: (metrics: any) => boolean;
  severity: Alert['severity'];
  component: string;
  message: string;
  cooldownMs: number;
  enabled: boolean;
}

export interface AlertChannel {
  name: string;
  type: 'console' | 'webhook' | 'email';
  config: Record<string, any>;
  enabled: boolean;
}

class AlertingSystem extends EventEmitter {
  private alerts: Map<string, Alert> = new Map();
  private rules: Map<string, AlertRule> = new Map();
  private channels: Map<string, AlertChannel> = new Map();
  private lastAlertTime: Map<string, number> = new Map();
  private checkInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.initializeDefaultRules();
    this.initializeDefaultChannels();
    this.startMonitoring();
  }

  private initializeDefaultRules(): void {
    // System health rules
    this.addRule({
      id: 'python_server_down',
      name: 'Python Server Down',
      condition: (metrics) => metrics.python_server_status === 0,
      severity: 'critical',
      component: 'python_server',
      message: 'Python AI server is not responding',
      cooldownMs: 60000, // 1 minute
      enabled: true
    });

    this.addRule({
      id: 'database_down',
      name: 'Database Down',
      condition: (metrics) => metrics.database_status === 0,
      severity: 'critical',
      component: 'database',
      message: 'Database is not accessible',
      cooldownMs: 60000,
      enabled: true
    });

    this.addRule({
      id: 'high_memory_usage',
      name: 'High Memory Usage',
      condition: (metrics) => {
        const memUsage = metrics.memory_usage_bytes?.heap_used || 0;
        const memTotal = metrics.memory_usage_bytes?.heap_total || 1;
        return (memUsage / memTotal) > 0.9; // 90% memory usage
      },
      severity: 'high',
      component: 'system',
      message: 'Memory usage is critically high (>90%)',
      cooldownMs: 300000, // 5 minutes
      enabled: true
    });

    this.addRule({
      id: 'high_error_rate',
      name: 'High Error Rate',
      condition: (metrics) => {
        const errors = metrics.analysis_errors_total || 0;
        const total = metrics.analysis_requests_total || 1;
        return (errors / total) > 0.1; // 10% error rate
      },
      severity: 'high',
      component: 'analysis',
      message: 'Analysis error rate is above 10%',
      cooldownMs: 300000,
      enabled: true
    });

    this.addRule({
      id: 'long_queue_length',
      name: 'Long Analysis Queue',
      condition: (metrics) => (metrics.analysis_queue_length || 0) > 10,
      severity: 'medium',
      component: 'analysis',
      message: 'Analysis queue is backing up (>10 items)',
      cooldownMs: 180000, // 3 minutes
      enabled: true
    });

    this.addRule({
      id: 'slow_response_time',
      name: 'Slow Response Time',
      condition: (metrics) => {
        const avgResponseTime = metrics.http_request_duration_seconds || 0;
        return avgResponseTime > 5; // 5 seconds
      },
      severity: 'medium',
      component: 'api',
      message: 'Average API response time is above 5 seconds',
      cooldownMs: 300000,
      enabled: true
    });

    this.addRule({
      id: 'no_active_sessions',
      name: 'No Active Sessions',
      condition: (metrics) => {
        const uptime = metrics.system_uptime_seconds || 0;
        const sessions = metrics.active_sessions || 0;
        return uptime > 300 && sessions === 0; // No sessions after 5 minutes uptime
      },
      severity: 'low',
      component: 'auth',
      message: 'No active user sessions detected',
      cooldownMs: 600000, // 10 minutes
      enabled: true
    });
  }

  private initializeDefaultChannels(): void {
    // Console logging channel
    this.addChannel({
      name: 'console',
      type: 'console',
      config: {},
      enabled: true
    });

    // Webhook channel (disabled by default)
    this.addChannel({
      name: 'webhook',
      type: 'webhook',
      config: {
        url: process.env.ALERT_WEBHOOK_URL || '',
        timeout: 5000
      },
      enabled: !!process.env.ALERT_WEBHOOK_URL
    });
  }

  addRule(rule: AlertRule): void {
    this.rules.set(rule.id, rule);
    logger.info(`Alert rule added: ${rule.name}`);
  }

  removeRule(ruleId: string): void {
    this.rules.delete(ruleId);
    logger.info(`Alert rule removed: ${ruleId}`);
  }

  addChannel(channel: AlertChannel): void {
    this.channels.set(channel.name, channel);
    logger.info(`Alert channel added: ${channel.name}`);
  }

  removeChannel(channelName: string): void {
    this.channels.delete(channelName);
    logger.info(`Alert channel removed: ${channelName}`);
  }

  checkRules(metrics: any): void {
    for (const rule of Array.from(this.rules.values())) {
      if (!rule.enabled) continue;

      try {
        const shouldAlert = rule.condition(metrics);
        
        if (shouldAlert) {
          const lastAlert = this.lastAlertTime.get(rule.id) || 0;
          const now = Date.now();
          
          // Check cooldown
          if (now - lastAlert < rule.cooldownMs) {
            continue;
          }

          this.triggerAlert({
            id: `${rule.id}_${now}`,
            severity: rule.severity,
            title: rule.name,
            message: rule.message,
            component: rule.component,
            timestamp: new Date(),
            resolved: false,
            metadata: { metrics, ruleId: rule.id }
          });

          this.lastAlertTime.set(rule.id, now);
        }
      } catch (error) {
        logger.error(`Error checking alert rule ${rule.id}:`, error);
      }
    }
  }

  private triggerAlert(alert: Alert): void {
    this.alerts.set(alert.id, alert);
    
    logger.warn(`ALERT [${alert.severity.toUpperCase()}] ${alert.title}: ${alert.message}`, {
      alertId: alert.id,
      component: alert.component,
      metadata: alert.metadata
    });

    // Send to all enabled channels
    for (const channel of Array.from(this.channels.values())) {
      if (channel.enabled) {
        this.sendToChannel(alert, channel);
      }
    }

    // Emit event for other systems to listen
    this.emit('alert', alert);
  }

  private async sendToChannel(alert: Alert, channel: AlertChannel): Promise<void> {
    try {
      switch (channel.type) {
        case 'console':
          console.error(`🚨 ALERT: [${alert.severity.toUpperCase()}] ${alert.title}`);
          console.error(`   Component: ${alert.component}`);
          console.error(`   Message: ${alert.message}`);
          console.error(`   Time: ${alert.timestamp.toISOString()}`);
          break;

        case 'webhook':
          if (channel.config.url) {
            const response = await fetch(channel.config.url, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                alert,
                service: 'SatyaAI',
                environment: process.env.NODE_ENV || 'development'
              }),
              signal: AbortSignal.timeout(channel.config.timeout || 5000)
            });

            if (!response.ok) {
              throw new Error(`Webhook returned ${response.status}`);
            }
          }
          break;

        case 'email':
          // Email implementation would go here
          logger.info(`Email alert would be sent: ${alert.title}`);
          break;

        default:
          logger.warn(`Unknown alert channel type: ${channel.type}`);
      }
    } catch (error) {
      logger.error(`Failed to send alert to channel ${channel.name}:`, error);
    }
  }

  resolveAlert(alertId: string): void {
    const alert = this.alerts.get(alertId);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = new Date();
      
      logger.info(`Alert resolved: ${alert.title}`, { alertId });
      this.emit('alertResolved', alert);
    }
  }

  getActiveAlerts(): Alert[] {
    return Array.from(this.alerts.values()).filter(alert => !alert.resolved);
  }

  getAllAlerts(): Alert[] {
    return Array.from(this.alerts.values());
  }

  getAlertHistory(hours: number = 24): Alert[] {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    return Array.from(this.alerts.values())
      .filter(alert => alert.timestamp >= cutoff)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  private startMonitoring(): void {
    // Check rules every 30 seconds
    this.checkInterval = setInterval(() => {
      // This would be called with current metrics
      // For now, we'll emit an event that the metrics system can listen to
      this.emit('checkRules');
    }, 30000);

    logger.info('Alerting system monitoring started');
  }

  stop(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    logger.info('Alerting system stopped');
  }

  // Health check for the alerting system itself
  getSystemHealth(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    activeRules: number;
    activeChannels: number;
    activeAlerts: number;
    lastCheck: Date;
  } {
    const activeRules = Array.from(this.rules.values()).filter(r => r.enabled).length;
    const activeChannels = Array.from(this.channels.values()).filter(c => c.enabled).length;
    const activeAlerts = this.getActiveAlerts().length;

    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    if (activeChannels === 0) {
      status = 'degraded';
    }
    
    if (activeRules === 0) {
      status = 'unhealthy';
    }

    return {
      status,
      activeRules,
      activeChannels,
      activeAlerts,
      lastCheck: new Date()
    };
  }
}

// Singleton instance
const alertingSystem = new AlertingSystem();

export default alertingSystem;