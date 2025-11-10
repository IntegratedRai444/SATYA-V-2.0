// Setup file for Jest tests
import '@testing-library/jest-dom';
import { TextEncoder, TextDecoder } from 'util';

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock ResizeObserver
class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = ResizeObserver;

// Mock window.scrollTo
window.scrollTo = jest.fn();

// Add TextEncoder/TextDecoder for node environment
if (typeof global.TextEncoder === 'undefined') {
  global.TextEncoder = TextEncoder;
}

if (typeof global.TextDecoder === 'undefined') {
  global.TextDecoder = TextDecoder as any;
}

// Mock console.error to catch React warnings
global.console.error = (...args) => {
  // Suppress specific warnings if needed
  const suppressedErrors = [
    'Warning: ReactDOM.render is no longer supported in React 18',
    'Warning: useLayoutEffect does nothing on the server',
  ];

  if (!suppressedErrors.some(entry => args[0].includes(entry))) {
    console.error(...args);
  }
};
