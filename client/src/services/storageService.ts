/**
 * Centralized Storage Service
 * Eliminates localStorage duplication across:
 * - AppContext.tsx
 * - auth.ts
 * - useLocalStorage.ts
 * - useSettings.ts
 * - Plus 2+ more files
 */

export class StorageService {
    /**
     * Get item from localStorage with automatic JSON parsing
     */
    static get<T>(key: string, defaultValue?: T): T | null {
        try {
            const item = localStorage.getItem(key);
            if (item === null) {
                return defaultValue !== undefined ? defaultValue : null;
            }

            // Try to parse as JSON, if it fails return as string
            try {
                return JSON.parse(item) as T;
            } catch {
                // If parsing fails, return as-is (for simple strings)
                return item as unknown as T;
            }
        } catch (error) {
            console.error(`Error reading from localStorage (key: ${key}):`, error);
            return defaultValue !== undefined ? defaultValue : null;
        }
    }

    /**
     * Set item in localStorage with automatic JSON stringification
     */
    static set<T>(key: string, value: T): boolean {
        try {
            const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(key, stringValue);
            return true;
        } catch (error) {
            console.error(`Error writing to localStorage (key: ${key}):`, error);
            return false;
        }
    }

    /**
     * Remove item from localStorage
     */
    static remove(key: string): boolean {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error(`Error removing from localStorage (key: ${key}):`, error);
            return false;
        }
    }

    /**
     * Clear all items from localStorage
     */
    static clear(): boolean {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('Error clearing localStorage:', error);
            return false;
        }
    }

    /**
     * Check if key exists in localStorage
     */
    static has(key: string): boolean {
        return localStorage.getItem(key) !== null;
    }

    /**
     * Get all keys from localStorage
     */
    static keys(): string[] {
        return Object.keys(localStorage);
    }

    /**
     * Get item with expiry support
     */
    static getWithExpiry<T>(key: string, defaultValue?: T): T | null {
        try {
            const itemStr = localStorage.getItem(key);
            if (!itemStr) {
                return defaultValue !== undefined ? defaultValue : null;
            }

            const item = JSON.parse(itemStr);
            const now = new Date().getTime();

            // Check if item has expired
            if (item.expiry && now > item.expiry) {
                localStorage.removeItem(key);
                return defaultValue !== undefined ? defaultValue : null;
            }

            return item.value as T;
        } catch (error) {
            console.error(`Error reading from localStorage with expiry (key: ${key}):`, error);
            return defaultValue !== undefined ? defaultValue : null;
        }
    }

    /**
     * Set item with expiry (in milliseconds)
     */
    static setWithExpiry<T>(key: string, value: T, ttl: number): boolean {
        try {
            const now = new Date().getTime();
            const item = {
                value,
                expiry: now + ttl
            };
            localStorage.setItem(key, JSON.stringify(item));
            return true;
        } catch (error) {
            console.error(`Error writing to localStorage with expiry (key: ${key}):`, error);
            return false;
        }
    }

    /**
     * Subscribe to storage changes (from other tabs)
     */
    static subscribe(key: string, callback: (newValue: any, oldValue: any) => void): () => void {
        const handler = (e: StorageEvent) => {
            if (e.key === key) {
                const newValue = e.newValue ? JSON.parse(e.newValue) : null;
                const oldValue = e.oldValue ? JSON.parse(e.oldValue) : null;
                callback(newValue, oldValue);
            }
        };

        window.addEventListener('storage', handler);

        // Return unsubscribe function
        return () => {
            window.removeEventListener('storage', handler);
        };
    }
}

// Export convenience functions
export const storage = {
    get: StorageService.get.bind(StorageService),
    set: StorageService.set.bind(StorageService),
    remove: StorageService.remove.bind(StorageService),
    clear: StorageService.clear.bind(StorageService),
    has: StorageService.has.bind(StorageService),
    keys: StorageService.keys.bind(StorageService),
    getWithExpiry: StorageService.getWithExpiry.bind(StorageService),
    setWithExpiry: StorageService.setWithExpiry.bind(StorageService),
    subscribe: StorageService.subscribe.bind(StorageService),
};
