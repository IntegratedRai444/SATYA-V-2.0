/**
 * Lazy loading utilities for performance optimization
 */

import { lazy } from 'react';

// Lazy load heavy components
export const LazyImageAnalysis = lazy(() => import('../pages/ImageAnalysis'));
export const LazyVideoAnalysis = lazy(() => import('../pages/VideoAnalysis'));
export const LazyAudioAnalysis = lazy(() => import('../pages/AudioAnalysis'));
export const LazyAnalytics = lazy(() => import('../pages/Analytics'));
export const LazySettings = lazy(() => import('../pages/Settings'));

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
    case 'style':
      link.as = 'style';
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
    console.error(`Failed to load chunk: ${chunkName}`, error);
    throw error;
  }
};