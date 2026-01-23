/**
 * Standardized API error formatter for consistent UX
 */

export interface FormattedError {
  title: string;
  description: string;
  retryable: boolean;
  code: string;
}

export const formatApiError = (error: { response?: { status?: number; data?: Record<string, unknown> } }): FormattedError => {
  // Handle network errors
  if (!error.response) {
    return {
      title: 'Network Error',
      description: 'Unable to connect to the server. Please check your internet connection.',
      retryable: true,
      code: 'NETWORK_ERROR'
    };
  }

  const { status, data } = error.response;

  // Handle specific HTTP status codes
  switch (status) {
    case 401:
      return {
        title: 'Authentication Required',
        description: 'Please log in to continue.',
        retryable: false,
        code: 'AUTH_REQUIRED'
      };

    case 403:
      return {
        title: 'CSRF Token Invalid',
        description: 'Security token expired. Please refresh the page and try again.',
        retryable: false,
        code: 'CSRF_INVALID'
      };

    case 413:
      return {
        title: 'File Too Large',
        description: 'The file size exceeds the maximum allowed limit (50MB).',
        retryable: false,
        code: 'FILE_TOO_LARGE'
      };

    case 415:
      return {
        title: 'Unsupported File Type',
        description: 'This file format is not supported. Please use a valid image, audio, or video file.',
        retryable: false,
        code: 'UNSUPPORTED_TYPE'
      };

    case 429:
      return {
        title: 'Rate Limit Exceeded',
        description: 'Too many requests. Please wait a moment and try again.',
        retryable: true,
        code: 'RATE_LIMIT'
      };

    case 500:
      return {
        title: 'Server Error',
        description: 'Something went wrong on our end. Please try again later.',
        retryable: true,
        code: 'SERVER_ERROR'
      };

    case 502:
      return {
        title: 'AI Engine Unavailable',
        description: 'The AI analysis service is temporarily unavailable. Please try again later.',
        retryable: true,
        code: 'PYTHON_DOWN'
      };

    case 503:
      return {
        title: 'Service Unavailable',
        description: 'The service is temporarily unavailable. Please try again later.',
        retryable: true,
        code: 'SERVICE_UNAVAILABLE'
      };

    case 504:
      return {
        title: 'Analysis Timeout',
        description: 'The analysis took too long to complete. Please try with a smaller file.',
        retryable: true,
        code: 'ANALYSIS_TIMEOUT'
      };

    default:
      // Handle backend-specific error codes
      if (data?.code) {
        return {
          title: (data.error as string) || 'Analysis Error',
          description: (data.message as string) || 'An error occurred during analysis.',
          retryable: data.retryable !== false,
          code: data.code as string
        };
      }

      return {
        title: 'Unknown Error',
        description: 'An unexpected error occurred. Please try again.',
        retryable: true,
        code: 'UNKNOWN_ERROR'
      };
  }
};
