import { useState, useEffect, useCallback } from 'react';

export interface Settings {
  theme: 'light' | 'dark' | 'system';
  notifications: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
  language: string;
  timezone: string;
  dateFormat: string;
  timeFormat: '12h' | '24h';
  // Add more settings as needed
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    theme: 'system',
    notifications: {
      email: true,
      push: true,
      sound: true,
    },
    language: 'en-US',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '12h',
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchSettings = useCallback(async () => {
    try {
      setIsLoading(true);
      // Replace with your actual API call to get user settings
      // const response = await fetch('/api/settings');
      // const data = await response.json();
      // setSettings(data.settings);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch settings'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateSettings = useCallback(async (updates: Partial<Settings>) => {
    try {
      setIsLoading(true);
      // Replace with your actual API call to update settings
      // const response = await fetch('/api/settings', {
      //   method: 'PATCH',
      //   body: JSON.stringify(updates),
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      // });
      // const data = await response.json();
      setSettings(prev => ({ ...prev, ...updates }));
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to update settings'));
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const resetToDefaults = useCallback(async () => {
    try {
      setIsLoading(true);
      // Replace with your actual API call to reset settings
      // await fetch('/api/settings/reset', { method: 'POST' });
      setSettings({
        theme: 'system',
        notifications: {
          email: true,
          push: true,
          sound: true,
        },
        language: 'en-US',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        dateFormat: 'MM/dd/yyyy',
        timeFormat: '12h',
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to reset settings'));
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  return {
    settings,
    isLoading,
    error,
    updateSettings,
    resetToDefaults,
    refetch: fetchSettings,
  };
}
