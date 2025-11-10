// Server configuration management
export interface ServerConfig {
  server_url: string;
  port: number;
  timestamp: string;
}

// Default server URL (fallback)
// Updated to match backend port for all API calls
const DEFAULT_SERVER_URL = 'http://localhost:5000';

// Cache for server config
let serverConfigCache: ServerConfig | null = null;

/**
 * Get the server configuration
 * Tries to read from server_config.json first, then falls back to default
 */
export async function getServerConfig(): Promise<ServerConfig> {
  if (serverConfigCache) {
    return serverConfigCache;
  }

  try {
    // Try to fetch the server config from the backend
    const response = await fetch('/server_config.json');
    if (response.ok) {
      const config = await response.json();
      serverConfigCache = config;
      return config;
    }
  } catch (error) {
    console.warn('Could not load server config, using default:', error);
  }

  // Fallback to default config
  const defaultConfig: ServerConfig = {
    server_url: DEFAULT_SERVER_URL,
    port: 5000,
    timestamp: new Date().toISOString()
  };

  serverConfigCache = defaultConfig;
  return defaultConfig;
}

/**
 * Get the base server URL for API calls
 */
export async function getServerUrl(): Promise<string> {
  const config = await getServerConfig();
  return config.server_url;
}

/**
 * Clear the server config cache (useful for development)
 */
export function clearServerConfigCache(): void {
  serverConfigCache = null;
}

/**
 * Create a full API URL
 */
export async function createApiUrl(endpoint: string): Promise<string> {
  const serverUrl = await getServerUrl();
  return `${serverUrl}${endpoint}`;
} 