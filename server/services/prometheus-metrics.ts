import type { Request, Response } from 'express';
import { performance } from 'perf_hooks';

interface MetricValue {
  value: number;
  timestamp: number;
  labels?: Record<string, string>;
}

interface Metric {
  name: string;
  help: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  values: Map<string, MetricValue>;
}

class PrometheusMetrics {
  private metrics: Map<string, Metric> = new Map();
  private startTime: number = Date.now();

  constructor() {
    this.initializeDefaultMetrics();
  }

  private initializeDefaultMetrics(): void {
    // HTTP request metrics
    this.registerMetric('http_requests_total', 'counter', 'Total number of HTTP requests');
    this.registerMetric('http_request_duration_seconds', 'histogram', 'HTTP request duration in seconds');
    this.registerMetric('http_requests_in_flight', 'gauge', 'Current number of HTTP requests being processed');

    // Analysis metrics
    this.registerMetric('analysis_requests_total', 'counter', 'Total number of analysis requests');
    this.registerMetric('analysis_duration_seconds', 'histogram', 'Analysis processing duration in seconds');
    this.registerMetric('analysis_errors_total', 'counter', 'Total number of analysis errors');
    this.registerMetric('analysis_queue_length', 'gauge', 'Current analysis queue length');

    // System metrics
    this.registerMetric('system_uptime_seconds', 'gauge', 'System uptime in seconds');
    this.registerMetric('active_websocket_connections', 'gauge', 'Number of active WebSocket connections');
    this.registerMetric('active_sessions', 'gauge', 'Number of active user sessions');
    this.registerMetric('memory_usage_bytes', 'gauge', 'Memory usage in bytes');

    // Health metrics
    this.registerMetric('health_check_status', 'gauge', 'Health check status (1=healthy, 0=unhealthy)');
    this.registerMetric('python_server_status', 'gauge', 'Python server status (1=up, 0=down)');
    this.registerMetric('database_status', 'gauge', 'Database status (1=up, 0=down)');
  }

  private registerMetric(name: string, type: Metric['type'], help: string): void {
    this.metrics.set(name, {
      name,
      help,
      type,
      values: new Map()
    });
  }

  // Counter methods
  incrementCounter(name: string, labels?: Record<string, string>, value: number = 1): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'counter') return;

    const labelKey = this.getLabelKey(labels);
    const existing = metric.values.get(labelKey);
    const newValue = (existing?.value || 0) + value;

    metric.values.set(labelKey, {
      value: newValue,
      timestamp: Date.now(),
      labels
    });
  }

  // Gauge methods
  setGauge(name: string, value: number, labels?: Record<string, string>): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'gauge') return;

    const labelKey = this.getLabelKey(labels);
    metric.values.set(labelKey, {
      value,
      timestamp: Date.now(),
      labels
    });
  }

  incrementGauge(name: string, labels?: Record<string, string>, value: number = 1): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'gauge') return;

    const labelKey = this.getLabelKey(labels);
    const existing = metric.values.get(labelKey);
    const newValue = (existing?.value || 0) + value;

    metric.values.set(labelKey, {
      value: newValue,
      timestamp: Date.now(),
      labels
    });
  }

  decrementGauge(name: string, labels?: Record<string, string>, value: number = 1): void {
    this.incrementGauge(name, labels, -value);
  }

  // Histogram methods (simplified - just track duration)
  recordHistogram(name: string, value: number, labels?: Record<string, string>): void {
    const metric = this.metrics.get(name);
    if (!metric || metric.type !== 'histogram') return;

    const labelKey = this.getLabelKey(labels);
    metric.values.set(labelKey, {
      value,
      timestamp: Date.now(),
      labels
    });
  }

  // Timing helper
  startTimer(name: string, labels?: Record<string, string>): () => void {
    const start = performance.now();
    return () => {
      const duration = (performance.now() - start) / 1000; // Convert to seconds
      this.recordHistogram(name, duration, labels);
    };
  }

  // Update system metrics
  updateSystemMetrics(): void {
    const uptime = (Date.now() - this.startTime) / 1000;
    this.setGauge('system_uptime_seconds', uptime);

    // Memory usage
    const memUsage = process.memoryUsage();
    this.setGauge('memory_usage_bytes', memUsage.heapUsed, { type: 'heap_used' });
    this.setGauge('memory_usage_bytes', memUsage.heapTotal, { type: 'heap_total' });
    this.setGauge('memory_usage_bytes', memUsage.rss, { type: 'rss' });
  }

  // Generate Prometheus format output
  generatePrometheusFormat(): string {
    let output = '';

    for (const metric of Array.from(this.metrics.values())) {
      // Add help comment
      output += `# HELP ${metric.name} ${metric.help}\n`;
      output += `# TYPE ${metric.name} ${metric.type}\n`;

      // Add metric values
      for (const [labelKey, metricValue] of Array.from(metric.values)) {
        const labelsStr = metricValue.labels
          ? Object.entries(metricValue.labels)
            .map(([key, value]) => `${key}="${value}"`)
            .join(',')
          : '';

        const metricLine = labelsStr
          ? `${metric.name}{${labelsStr}} ${metricValue.value}`
          : `${metric.name} ${metricValue.value}`;

        output += `${metricLine}\n`;
      }
      output += '\n';
    }

    return output;
  }

  // Express middleware for HTTP metrics
  httpMetricsMiddleware(): (req: Request, res: Response, next: () => void) => void {
    return (req: Request, res: Response, next: () => void) => {
      const start = process.hrtime();
      const path = req.path;
      const method = req.method;

      res.on('finish', () => {
        const duration = process.hrtime(start);
        const responseTime = duration[0] * 1e3 + duration[1] * 1e-6; // Convert to milliseconds
        
        this.incrementCounter('http_requests_total', { method, path, status: res.statusCode.toString() });
        this.recordHistogram('http_request_duration_seconds', responseTime / 1000, { method, path });
      });

      next();
    };
  }

  recordAnalysisStart(type: string): () => void {
    this.incrementCounter('analysis_requests_total', { type });
    this.incrementGauge('analysis_queue_length');
    return this.startTimer('analysis_duration_seconds', { type });
  }

  recordAnalysisComplete(type: string): void {
    this.decrementGauge('analysis_queue_length');
  }

  recordAnalysisError(type: string, error: string): void {
    this.incrementCounter('analysis_errors_total', { type, error });
    this.decrementGauge('analysis_queue_length');
  }

  updateHealthStatus(component: string, status: boolean): void {
    this.setGauge('health_check_status', status ? 1 : 0, { component });
  }

  recordWebSocketConnection(connected: boolean): void {
    if(connected) {
      this.incrementGauge('active_websocket_connections');
    } else {
      this.decrementGauge('active_websocket_connections');
    }
  }

  updateActiveSessionsCount(count: number): void {
    this.setGauge('active_sessions', count);
  }

  private getLabelKey(labels?: Record<string, string>): string {
    if (!labels) return '';
    return Object.entries(labels)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${key}:${value}`)
      .join('|');
  }
}

// Singleton instance
const metrics = new PrometheusMetrics();

// Update system metrics every 30 seconds
setInterval(() => {
  metrics.updateSystemMetrics();
}, 30000);

export default metrics;