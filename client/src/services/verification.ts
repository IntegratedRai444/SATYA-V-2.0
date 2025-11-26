import apiClient from '../lib/api';
import logger from '../lib/logger';

/**
 * Test the full authentication flow from the frontend
 */
export const testAuthFlow = async (): Promise<{ success: boolean; logs: string[] }> => {
    const logs: string[] = [];
    const log = (msg: string) => {
        logs.push(msg);
        logger.info(`[AuthTest] ${msg}`);
    };

    log('Starting Auth Flow Verification...');

    try {
        // 1. Register a test user
        const timestamp = Date.now();
        const username = `test_fe_${timestamp}`;
        const email = `${username}@example.com`;
        const password = 'TestPassword123!';

        log(`1. Registering user: ${username}...`);
        const regRes = await apiClient.register(username, email, password);

        if (!regRes.success || !regRes.token) {
            log(`❌ Registration failed: ${regRes.message}`);
            return { success: false, logs };
        }
        log('✅ Registration successful');

        // 2. Login (implicitly done by register, but let's test explicit login)
        // First logout to clear the session from register
        log('2. Logging out to test explicit login...');
        await apiClient.logout();
        apiClient.clearAuth(); // Ensure local state is cleared

        log(`3. Logging in user: ${username}...`);
        const loginRes = await apiClient.login(username, password);

        if (!loginRes.success || !loginRes.token) {
            log(`❌ Login failed: ${loginRes.message}`);
            return { success: false, logs };
        }
        log('✅ Login successful');

        // 3. Validate Session
        log('4. Validating session (/api/auth/me)...');
        const sessionRes = await apiClient.validateSession();
        if (!sessionRes.valid || sessionRes.user?.username !== username) {
            log(`❌ Session validation failed or user mismatch`);
            return { success: false, logs };
        }
        // 5. Final Logout
        log('6. Final logout...');
        await apiClient.logout();
        log('✅ Logout successful');

        log('🎉 Auth Flow Verification PASSED');
        return { success: true, logs };

    } catch (error: any) {
        log(`❌ Exception during test: ${error.message}`);
        return { success: false, logs };
    }
};

/**
 * Fetch backend routes for verification
 */
export const fetchBackendRoutes = async (): Promise<any[]> => {
    try {
        const response = await apiClient.client.get('/health/wiring');
        return response.data.routes || [];
    } catch (error) {
        logger.error('Failed to fetch backend routes', error as Error);
        return [];
    }
};
