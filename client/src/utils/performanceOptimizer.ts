/**
 * Performance optimization utilities
 */

/**
 * Debounce function calls
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Throttle function calls
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Memoization for expensive calculations
 */
export function memoize<T extends (...args: any[]) => any>(
  func: T,
  getKey?: (...args: Parameters<T>) => string
): T {
  const cache = new Map<string, ReturnType<T>>();
  
  return ((...args: Parameters<T>) => {
    const key = getKey ? getKey(...args) : JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key);
    }
    
    const result = func(...args);
    cache.set(key, result);
    
    return result;
  }) as T;
}

/**
 * Virtual scrolling for large lists
 */
export class VirtualScroller {
  private container: HTMLElement;
  private itemHeight: number;
  private visibleCount: number;
  private totalCount: number;
  private scrollTop = 0;
  private renderCallback: (startIndex: number, endIndex: number) => void;

  constructor(
    container: HTMLElement,
    itemHeight: number,
    totalCount: number,
    renderCallback: (startIndex: number, endIndex: number) => void
  ) {
    this.container = container;
    this.itemHeight = itemHeight;
    this.totalCount = totalCount;
    this.renderCallback = renderCallback;
    this.visibleCount = Math.ceil(container.clientHeight / itemHeight) + 2; // Buffer

    this.setupScrollListener();
    this.updateVisibleItems();
  }

  private setupScrollListener() {
    const throttledScroll = throttle(() => {
      this.scrollTop = this.container.scrollTop;
      this.updateVisibleItems();
    }, 16); // ~60fps

    this.container.addEventListener('scroll', throttledScroll);
  }

  private updateVisibleItems() {
    const startIndex = Math.floor(this.scrollTop / this.itemHeight);
    const endIndex = Math.min(startIndex + this.visibleCount, this.totalCount);
    
    this.renderCallback(startIndex, endIndex);
  }

  updateTotalCount(count: number) {
    this.totalCount = count;
    this.updateVisibleItems();
  }
}

/**
 * Image optimization utilities
 */
export class ImageOptimizer {
  /**
   * Create responsive image srcset
   */
  static createSrcSet(baseUrl: string, sizes: number[]): string {
    return sizes
      .map(size => `${baseUrl}?w=${size} ${size}w`)
      .join(', ');
  }

  /**
   * Get optimal image size based on container
   */
  static getOptimalSize(containerWidth: number, devicePixelRatio = 1): number {
    const targetWidth = containerWidth * devicePixelRatio;
    const sizes = [320, 640, 768, 1024, 1280, 1920, 2560];
    
    return sizes.find(size => size >= targetWidth) || sizes[sizes.length - 1];
  }

  /**
   * Preload critical images
   */
  static preloadImages(urls: string[]): Promise<void[]> {
    return Promise.all(
      urls.map(url => 
        new Promise<void>((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve();
          img.onerror = reject;
          img.src = url;
        })
      )
    );
  }
}

/**
 * Bundle size analyzer
 */
export class BundleAnalyzer {
  private static loadTimes = new Map<string, number>();

  static trackChunkLoad(chunkName: string, startTime: number) {
    const loadTime = performance.now() - startTime;
    this.loadTimes.set(chunkName, loadTime);
    
    // Log slow chunks
    if (loadTime > 1000) {
      console.warn(`Slow chunk load: ${chunkName} took ${loadTime.toFixed(2)}ms`);
    }
  }

  static getLoadTimes(): Record<string, number> {
    return Object.fromEntries(this.loadTimes);
  }

  static getAverageLoadTime(): number {
    const times = Array.from(this.loadTimes.values());
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  }
}

/**
 * Performance monitoring
 */
export class PerformanceMonitor {
  private static metrics = new Map<string, number[]>();

  static mark(name: string) {
    performance.mark(name);
  }

  static measure(name: string, startMark: string, endMark?: string) {
    if (endMark) {
      performance.measure(name, startMark, endMark);
    } else {
      performance.measure(name, startMark);
    }

    const measure = performance.getEntriesByName(name, 'measure')[0];
    if (measure) {
      const existing = this.metrics.get(name) || [];
      existing.push(measure.duration);
      this.metrics.set(name, existing);
    }
  }

  static getMetrics(): Record<string, { avg: number; min: number; max: number; count: number }> {
    const result: Record<string, any> = {};
    
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
  static getMemoryUsage(): any {
    if ('memory' in performance) {
      return {
        used: Math.round((performance as any).memory.usedJSHeapSize / 1048576),
        total: Math.round((performance as any).memory.totalJSHeapSize / 1048576),
        limit: Math.round((performance as any).memory.jsHeapSizeLimit / 1048576)
      };
    }
    return null;
  }

  static logMemoryUsage(label: string) {
    const memory = this.getMemoryUsage();
    if (memory) {
      console.log(`${label} - Memory: ${memory.used}MB / ${memory.total}MB (limit: ${memory.limit}MB)`);
    }
  }
}