/**
 * Web Vitals Performance Monitoring
 * Tracks Core Web Vitals and reports to analytics
 */

import logger from './logger';

export interface WebVitalsMetric {
  name: 'CLS' | 'FID' | 'FCP' | 'LCP' | 'TTFB' | 'INP';
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  id: string;
  navigationType: string;
}

/**
 * Report Web Vitals to analytics
 */
function reportWebVitals(metric: WebVitalsMetric) {
  logger.info(`Web Vital: ${metric.name}`, {
    value: metric.value,
    rating: metric.rating,
    id: metric.id,
  });

  // Send to analytics service
  if (window.gtag) {
    window.gtag('event', metric.name, {
      value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
      event_category: 'Web Vitals',
      event_label: metric.id,
      non_interaction: true,
    });
  }
}

/**
 * Initialize Web Vitals monitoring
 */
export async function initWebVitals() {
  if (import.meta.env.PROD) {
    try {
      const { onCLS, onTTFB, onFCP, onLCP, onINP } = await import('web-vitals');
      
      onCLS(reportWebVitals);
      onTTFB(reportWebVitals);
      onFCP(reportWebVitals);
      onLCP(reportWebVitals);
      onTTFB(reportWebVitals);
      onINP(reportWebVitals);
      
      logger.info('Web Vitals monitoring initialized');
    } catch (error) {
      logger.error('Failed to initialize Web Vitals', error as Error);
    }
  }
}

export default initWebVitals;
