import logger from '../lib/logger';
import { notificationService } from '../services/notificationService';

export interface ApiError extends Error {
    status?: number;
    code?: string;
    response?: any;
}

export class ApiErrorHandler {
    /**
     * Handle API errors with consistent logging and user notification
     */
    static handle(error: any, context?: string): ApiError {
        const apiError = this.normalizeError(error);

        // Log the error
        logger.error(context || 'API Error', apiError, {
            status: apiError.status,
            code: apiError.code,
            response: apiError.response
        });

        // Show user-friendly notification based on error type
        this.notifyUser(apiError, context);

        return apiError;
    }

    /**
     * Normalize different error formats into ApiError
     */
    private static normalizeError(error: any): ApiError {
        if (error.response) {
            // Axios error with response
            const apiError: ApiError = new Error(
                error.response.data?.message ||
                error.response.data?.error ||
                error.message ||
                'Request failed'
            );
            apiError.status = error.response.status;
            apiError.code = error.response.data?.code || error.code;
            apiError.response = error.response.data;
            return apiError;
        }

        if (error.request) {
            // Network error (no response)
            const apiError: ApiError = new Error('Network error: Unable to reach server');
            apiError.code = 'NETWORK_ERROR';
            return apiError;
        }

        // Generic error
        const apiError: ApiError = error instanceof Error ? error : new Error(String(error));
        return apiError;
    }

    /**
     * Show user-friendly notifications based on error type
     */
    private static notifyUser(error: ApiError, context?: string): void {
        const status = error.status;

        // Don't show notifications for certain status codes
        if (status === 401) {
            // Auth errors are handled by AuthContext
            return;
        }

        let title = 'Error';
        let message = error.message;

        // Customize message based on status code
        switch (status) {
            case 400:
                title = 'Invalid Request';
                message = error.message || 'Please check your input and try again';
                break;
            case 403:
                title = 'Access Denied';
                message = 'You don\'t have permission to perform this action';
                break;
            case 404:
                title = 'Not Found';
                message = 'The requested resource was not found';
                break;
            case 409:
                title = 'Conflict';
                message = error.message || 'This action conflicts with existing data';
                break;
            case 429:
                title = 'Too Many Requests';
                message = 'Please slow down and try again in a moment';
                break;
            case 500:
            case 502:
            case 503:
                title = 'Server Error';
                message = 'Something went wrong on our end. Please try again later';
                break;
            default:
                if (error.code === 'NETWORK_ERROR') {
                    title = 'Connection Error';
                    message = 'Unable to connect to server. Please check your internet connection';
                } else if (context) {
                    title = `${context} Failed`;
                }
        }

        // Show notification
        notificationService.error(title, message, {
            duration: 5000,
            data: { error }
        });
    }

    /**
     * Check if error is a network error
     */
    static isNetworkError(error: any): boolean {
        return error.code === 'NETWORK_ERROR' ||
            error.message?.includes('Network Error') ||
            error.code === 'ECONNREFUSED';
    }

    /**
     * Check if error is an authentication error
     */
    static isAuthError(error: any): boolean {
        return error.status === 401 || error.status === 403;
    }

    /**
     * Check if error is a validation error
     */
    static isValidationError(error: any): boolean {
        return error.status === 400 || error.status === 422;
    }

    /**
     * Extract validation errors from response
     */
    static getValidationErrors(error: any): Record<string, string[]> | null {
        if (!this.isValidationError(error)) {
            return null;
        }

        return error.response?.errors || error.response?.validationErrors || null;
    }
}

/**
 * Axios interceptor for automatic error handling
 * Add this to your axios instance:
 * 
 * apiClient.interceptors.response.use(
 *   response => response,
 *   error => {
 *     ApiErrorHandler.handle(error);
 *     return Promise.reject(error);
 *   }
 * );
 */
export function createErrorInterceptor(context?: string) {
    return (error: any) => {
        const handledError = ApiErrorHandler.handle(error, context);
        return Promise.reject(handledError);
    };
}
