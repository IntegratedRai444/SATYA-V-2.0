/**
 * Accessibility Utilities
 * Provides utilities for WCAG AA compliance and screen reader support
 */

import logger from './logger';

// ============================================================================
// ARIA Live Region Manager
// ============================================================================

class AriaLiveRegionManager {
  private politeRegion: HTMLDivElement | null = null;
  private assertiveRegion: HTMLDivElement | null = null;

  constructor() {
    this.createLiveRegions();
  }

  private createLiveRegions() {
    // Create polite live region
    this.politeRegion = document.createElement('div');
    this.politeRegion.setAttribute('aria-live', 'polite');
    this.politeRegion.setAttribute('aria-atomic', 'true');
    this.politeRegion.className = 'sr-only';
    document.body.appendChild(this.politeRegion);

    // Create assertive live region
    this.assertiveRegion = document.createElement('div');
    this.assertiveRegion.setAttribute('aria-live', 'assertive');
    this.assertiveRegion.setAttribute('aria-atomic', 'true');
    this.assertiveRegion.className = 'sr-only';
    document.body.appendChild(this.assertiveRegion);
  }

  /**
   * Announce message to screen readers
   */
  announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    const region = priority === 'assertive' ? this.assertiveRegion : this.politeRegion;
    
    if (region) {
      // Clear and set new message
      region.textContent = '';
      setTimeout(() => {
        region.textContent = message;
      }, 100);
      
      logger.debug('Screen reader announcement', { message, priority });
    }
  }

  /**
   * Clear announcements
   */
  clear() {
    if (this.politeRegion) this.politeRegion.textContent = '';
    if (this.assertiveRegion) this.assertiveRegion.textContent = '';
  }
}

// ============================================================================
// Focus Management
// ============================================================================

/**
 * Trap focus within a container
 */
export function trapFocus(container: HTMLElement): () => void {
  const focusableElements = container.querySelectorAll<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  // Focus first element
  firstElement?.focus();

  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', handleKeyDown);
  };
}

/**
 * Manage focus on route change
 */
export function manageFocusOnRouteChange(targetId: string = 'main-content') {
  const target = document.getElementById(targetId);
  if (target) {
    target.setAttribute('tabindex', '-1');
    target.focus();
    target.removeAttribute('tabindex');
  }
}

/**
 * Get all focusable elements in container
 */
export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const selector = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  return Array.from(container.querySelectorAll<HTMLElement>(selector));
}

// ============================================================================
// Color Contrast Checker
// ============================================================================

/**
 * Check if color contrast meets WCAG AA standards
 */
export function checkColorContrast(
  foreground: string,
  background: string,
  fontSize: number = 16
): { ratio: number; passes: boolean; level: 'AAA' | 'AA' | 'fail' } {
  const fgLuminance = getRelativeLuminance(foreground);
  const bgLuminance = getRelativeLuminance(background);
  
  const ratio = (Math.max(fgLuminance, bgLuminance) + 0.05) / 
                (Math.min(fgLuminance, bgLuminance) + 0.05);

  // WCAG AA requires 4.5:1 for normal text, 3:1 for large text (18pt+)
  const isLargeText = fontSize >= 18;
  const requiredRatio = isLargeText ? 3 : 4.5;
  const aaaRatio = isLargeText ? 4.5 : 7;

  return {
    ratio: Math.round(ratio * 100) / 100,
    passes: ratio >= requiredRatio,
    level: ratio >= aaaRatio ? 'AAA' : ratio >= requiredRatio ? 'AA' : 'fail',
  };
}

/**
 * Get relative luminance of a color
 */
function getRelativeLuminance(color: string): number {
  const rgb = hexToRgb(color);
  if (!rgb) return 0;

  const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(val => {
    const normalized = val / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : Math.pow((normalized + 0.055) / 1.055, 2.4);
  });

  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Convert hex color to RGB
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

/**
 * Handle keyboard navigation for lists
 */
export function handleListKeyboardNavigation(
  event: React.KeyboardEvent,
  currentIndex: number,
  itemCount: number,
  onSelect: (index: number) => void
): void {
  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault();
      onSelect(Math.min(currentIndex + 1, itemCount - 1));
      break;
    case 'ArrowUp':
      event.preventDefault();
      onSelect(Math.max(currentIndex - 1, 0));
      break;
    case 'Home':
      event.preventDefault();
      onSelect(0);
      break;
    case 'End':
      event.preventDefault();
      onSelect(itemCount - 1);
      break;
    case 'Enter':
    case ' ':
      event.preventDefault();
      // Trigger selection
      break;
  }
}

// ============================================================================
// Skip Link Component Helper
// ============================================================================

export interface SkipLinkConfig {
  targetId: string;
  label: string;
}

export const defaultSkipLinks: SkipLinkConfig[] = [
  { targetId: 'main-content', label: 'Skip to main content' },
  { targetId: 'navigation', label: 'Skip to navigation' },
  { targetId: 'footer', label: 'Skip to footer' },
];

// ============================================================================
// Singleton Instance
// ============================================================================

export const ariaLiveRegion = new AriaLiveRegionManager();

// ============================================================================
// Exported Functions
// ============================================================================

export function announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
  ariaLiveRegion.announce(message, priority);
}

export function clearAnnouncements() {
  ariaLiveRegion.clear();
}

export default {
  announce,
  clearAnnouncements,
  trapFocus,
  manageFocusOnRouteChange,
  getFocusableElements,
  checkColorContrast,
  handleListKeyboardNavigation,
  defaultSkipLinks,
};
