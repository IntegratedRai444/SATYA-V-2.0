import apiClient from '../lib/api';
import logger from '../lib/logger';

/**
 * Check if user is properly authenticated
 */
export const testAuthFlow = async () => {
    const logs: string[] = [];
    const log = (message: string) => {
        logs.push(`[${new Date().toISOString()}] ${message}`);
        console.log(message);
    };

    log('Starting Auth Flow Verification...');

    try {
        // Check if user is already authenticated
        const currentUser = await apiClient.get('/auth/me') as { success: boolean; user?: unknown };
        if (currentUser && currentUser.success) {
            log('✅ User is already authenticated');
            return { success: true, logs, user: currentUser.user };
        }

        log('❌ No active session found');
        log('Please login through the login page to proceed');
        return { success: false, logs };

    } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        log(`❌ Auth verification failed: ${errorMessage}`);
        return { success: false, logs, error: errorMessage };
    }
};

/**
 * Fetch backend routes for verification
 */
export const fetchBackendRoutes = async (): Promise<unknown[]> => {
    try {
        const response = await apiClient.get('/health/wiring') as { data?: unknown[] };
        return response.data || [];
    } catch (error) {
        logger.error('Failed to fetch backend routes', error as Error);
        return [];
    }
};
