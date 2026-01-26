/**
 * Lazy loading utilities for performance optimization
 * Enhanced with retry logic and preloading
 */

import React, { lazy, ComponentType } from 'react';
import logger from '../lib/logger';

// Add type for IntersectionObserverInit
interface IntersectionObserverInit {
  root?: Element | null;
  rootMargin?: string;
  threshold?: number | number[];
}

/**
 * Enhanced lazy loading with retry logic
 */
function lazyWithRetry<T extends ComponentType<Record<string, unknown>>>(
  componentImport: () => Promise<{ default: T }>,
  maxRetries = 3,
  retryDelay = 1000
): React.LazyExoticComponent<T> {
  return lazy(async () => {
    const pageHasAlreadyBeenForceRefreshed = JSON.parse(
      window.sessionStorage.getItem('page-has-been-force-refreshed') || 'false'
    );

    try {
      const component = await componentImport();
      window.sessionStorage.setItem('page-has-been-force-refreshed', 'false');
      return component;
    } catch (error) {
      logger.error('Failed to load component', error as Error);

      if (!pageHasAlreadyBeenForceRefreshed) {
        window.sessionStorage.setItem('page-has-been-force-refreshed', 'true');
        logger.info('Reloading page to fetch latest chunks');
        return window.location.reload() as never;
      }

      // Retry logic
      for (let i = 0; i < maxRetries; i++) {
        try {
          logger.info(`Retrying component load (attempt ${i + 1}/${maxRetries})`);
          await new Promise(resolve => setTimeout(resolve, retryDelay * (i + 1)));
          return await componentImport();
        } catch (retryError) {
          if (i === maxRetries - 1) {
            throw retryError;
          }
        }
      }

      throw error;
    }
  });
}

// Lazy load heavy components with retry
export const LazyImageAnalysis = lazyWithRetry(() => import('../pages/ImageAnalysis'));
export const LazyVideoAnalysis = lazyWithRetry(() => import('../pages/VideoAnalysis'));
export const LazyAudioAnalysis = lazyWithRetry(() => import('../pages/AudioAnalysis'));
export const LazyAnalytics = lazyWithRetry(() => import('../pages/Analytics'));
export const LazySettings = lazyWithRetry(() => import('../pages/Settings'));
export const LazyBatchAnalysis = lazyWithRetry(() => import('../pages/BatchAnalysis'));
export const LazyHistory = lazyWithRetry(() => import('../pages/History'));
export const LazyHelp = lazyWithRetry(() => import('../pages/Help'));
export const LazyAIAssistant = lazyWithRetry(() => import('../pages/AIAssistant'));

// Lazy load heavy analysis components
export const LazyAnalysisResults = lazy(() => import('../components/analysis/AnalysisResults'));
export const LazyAnalysisProgress = lazy(() => import('../components/analysis/AnalysisProgress'));

/**
 * Intersection Observer for lazy loading elements
 */
export class LazyLoader {
  private observer: IntersectionObserver;
  private loadedElements = new Set<Element>();

  constructor(options: IntersectionObserverInit = {}) {
    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        rootMargin: '50px',
        threshold: 0.1,
        ...options
      }
    );
  }

  private handleIntersection(entries: IntersectionObserverEntry[]) {
    entries.forEach(entry => {
      if (entry.isIntersecting && !this.loadedElements.has(entry.target)) {
        this.loadElement(entry.target);
        this.loadedElements.add(entry.target);
        this.observer.unobserve(entry.target);
      }
    });
  }

  private loadElement(element: Element) {
    // Load images
    if (element.tagName === 'IMG') {
      const img = element as HTMLImageElement;
      const dataSrc = img.dataset.src;
      if (dataSrc) {
        img.src = dataSrc;
        img.removeAttribute('data-src');
      }
    }

    // Load background images
    const dataBg = element.getAttribute('data-bg');
    if (dataBg) {
      (element as HTMLElement).style.backgroundImage = `url(${dataBg})`;
      element.removeAttribute('data-bg');
    }

    // Trigger custom load event
    const loadEvent = new CustomEvent('lazyload', { detail: { element } });
    element.dispatchEvent(loadEvent);
  }

  observe(element: Element) {
    this.observer.observe(element);
  }

  unobserve(element: Element) {
    this.observer.unobserve(element);
    this.loadedElements.delete(element);
  }

  disconnect() {
    this.observer.disconnect();
    this.loadedElements.clear();
  }
}

/**
 * Hook for lazy loading images
 */
export const useLazyImage = (src: string, placeholder?: string) => {
  const [imageSrc, setImageSrc] = React.useState(placeholder || '');
  const [isLoaded, setIsLoaded] = React.useState(false);
  const imgRef = React.useRef<HTMLImageElement>(null);

  React.useEffect(() => {
    const loader = new LazyLoader();
    const img = imgRef.current;

    if (img) {
      img.dataset.src = src;

      const handleLoad = () => {
        setImageSrc(src);
        setIsLoaded(true);
      };

      img.addEventListener('lazyload', handleLoad);
      loader.observe(img);

      return () => {
        img.removeEventListener('lazyload', handleLoad);
        loader.disconnect();
      };
    }
  }, [src]);

  return { imageSrc, isLoaded, imgRef };
};

/**
 * Preload critical resources
 */
export const preloadResource = (url: string, type: 'image' | 'script' | 'style' = 'image') => {
  const link = document.createElement('link');
  link.rel = 'preload';
  link.href = url;

  switch (type) {
    case 'image':
      link.as = 'image';
      break;
    case 'script':
      link.as = 'script';
      break;
  }

  document.head.appendChild(link);
};

/**
 * Bundle splitting helper
 */
export const loadChunk = async (chunkName: string) => {
  try {
    const module = await import(/* webpackChunkName: "[request]" */ `../chunks/${chunkName}`);
    return module.default || module;
  } catch (error) {
    logger.error(`Failed to load chunk: ${chunkName}`, error as Error);
    throw error;
  }
};

/**
 * Preload a lazy component
 */
export function preloadComponent<T extends ComponentType<unknown>>(
  LazyComponent: React.LazyExoticComponent<T>
): void {
  // @ts-expect-error - accessing internal preload method
  if (LazyComponent._payload && LazyComponent._payload._result === null) {
    // @ts-expect-error - accessing internal property
    LazyComponent._payload._result = LazyComponent._payload._fn();
  }
}

/**
 * Preload critical routes on idle
 */
export function preloadCriticalRoutes(): void {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      preloadComponent(LazyImageAnalysis);
      preloadComponent(LazyVideoAnalysis);
    });
  } else {
    setTimeout(() => {
      preloadComponent(LazyImageAnalysis);
      preloadComponent(LazyVideoAnalysis);
    }, 1000);
  }
}

export { lazyWithRetry };
export default lazyWithRetry;