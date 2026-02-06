/**
 * Performance optimization utilities
 */

/**
 * Performance monitoring and optimization
 */
export class PerformanceOptimizer {
  private static metrics = new Map<string, number[]>();

  /**
   * Start timing a performance metric
   */
  static startTiming(name: string): void {
    performance.mark(`${name}-start`);
  }

  /**
   * End timing a performance metric
   */
  static endTiming(name: string): number {
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);
    
    const entries = performance.getEntriesByName(name);
    if (entries.length > 0) {
      const duration = entries[entries.length - 1].duration;
      this.recordMetric(name, duration);
      return duration;
    }
    return 0;
  }

  /**
   * Record a performance metric
   */
  static recordMetric(name: string, duration: number): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);
  }

  /**
   * Get metrics summary
   */
  static getMetrics(): Record<string, { avg: number; min: number; max: number; count: number }> {
    const result: Record<string, { avg: number; min: number; max: number; count: number }> = {};
    
    for (const [name, durations] of this.metrics) {
      result[name] = {
        avg: durations.reduce((a, b) => a + b, 0) / durations.length,
        min: Math.min(...durations),
        max: Math.max(...durations),
        count: durations.length
      };
    }
    
    return result;
  }

  static clearMetrics() {
    this.metrics.clear();
    performance.clearMarks();
    performance.clearMeasures();
  }
}

/**
 * Memory usage monitoring
 */
export class MemoryMonitor {
  static getMemoryUsage(): { used: number; total: number; limit: number } | null {
    if ('memory' in performance) {
      return {
        used: Math.round((performance as Performance & { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1048576),
        total: Math.round((performance as Performance & { memory: { totalJSHeapSize: number } }).memory.totalJSHeapSize / 1048576),
        limit: Math.round((performance as Performance & { memory: { jsHeapSizeLimit: number } }).memory.jsHeapSizeLimit / 1048576)
      };
    }
    return null;
  }

  static logMemoryUsage() {
    const memory = this.getMemoryUsage();
    if (memory) {
      // Memory usage logged for debugging
    }
  }
}

/**
 * Performance monitoring for tracking app performance
 */
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, number[]> = new Map();
  private observers: PerformanceObserver[] = [];

  private constructor() {
    this.setupObservers();
  }

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  static mark(name: string): void {
    performance.mark(name);
  }

  private setupObservers(): void {
    if ('PerformanceObserver' in window) {
      // Observe navigation timing
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordMetric('navigation', entry.duration);
          }
        });
        observer.observe({ entryTypes: ['navigation'] });
        this.observers.push(observer);
      } catch {
        // Performance observer not supported
      }

      // Observe resource timing
      try {
        const resourceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordMetric('resource', entry.duration);
          }
        });
        resourceObserver.observe({ entryTypes: ['resource'] });
        this.observers.push(resourceObserver);
      } catch {
        // Resource observer not supported
      }
    }
  }

  recordMetric(name: string, duration: number): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);
  }

  getMetrics(): Record<string, { avg: number; min: number; max: number; count: number }> {
    const result: Record<string, { avg: number; min: number; max: number; count: number }> = {};
    
    for (const [name, durations] of this.metrics) {
      result[name] = {
        avg: durations.reduce((a, b) => a + b, 0) / durations.length,
        min: Math.min(...durations),
        max: Math.max(...durations),
        count: durations.length
      };
    }
    
    return result;
  }

  clearMetrics(): void {
    this.metrics.clear();
    performance.clearMarks();
    performance.clearMeasures();
  }

  getMemoryUsage(): { used: number; total: number; limit: number } | null {
    return MemoryMonitor.getMemoryUsage();
  }

  logMemoryUsage(): void {
    MemoryMonitor.logMemoryUsage();
  }

  disconnect(): void {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}