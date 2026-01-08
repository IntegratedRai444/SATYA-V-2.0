import * as Sentry from '@sentry/react';

type ErrorContext = {
  component?: string;
  action?: string;
  userId?: string;
  [key: string]: any;
};

export const logError = (
  error: Error | string,
  context: ErrorContext = {}
) => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorStack = typeof error === 'object' ? error.stack : undefined;
  
  // Log to console in development
  if (import.meta.env.DEV) {
    console.error('Error:', {
      message: errorMessage,
      stack: errorStack,
      context,
    });
  }

  // Send to Sentry in production
  if (import.meta.env.PROD) {
    Sentry.withScope((scope) => {
      scope.setExtras({
        ...context,
        stack: errorStack,
      });
      
      if (context.userId) {
        scope.setUser({ id: context.userId });
      }
      
      if (context.component) {
        scope.setTag('component', context.component);
      }
      
      Sentry.captureException(error);
    });
  }
};

// Example usage:
// try {
//   // Your code here
// } catch (error) {
//   logError(error, {
//     component: 'ComponentName',
//     action: 'fetchData',
//     userId: 'user123',
//     additionalInfo: 'More context here'
//   });
// }
