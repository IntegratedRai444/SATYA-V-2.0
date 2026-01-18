import { useState, useEffect, useCallback } from 'react';
import { User } from '@/types';

export function useUser() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchUser = useCallback(async () => {
    try {
      setIsLoading(true);
      // Replace with your actual API call to get the current user
      // const response = await fetch('/api/auth/me');
      // const data = await response.json();
      // setUser(data.user);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch user'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateUser = useCallback(async (updates: Partial<User>) => {
    try {
      setIsLoading(true);
      // Replace with your actual API call to update user
      console.log('Updating user with:', updates);
      // const response = await fetch('/api/auth/update', {
      //   method: 'PATCH',
      //   body: JSON.stringify(updates),
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      // });
      // const data = await response.json();
      // setUser(data.user);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to update user'));
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      // Replace with your actual logout logic
      // await fetch('/api/auth/logout', { method: 'POST' });
      setUser(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to log out'));
      throw err;
    }
  }, []);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  return {
    user,
    isLoading,
    error,
    updateUser,
    logout,
    refetch: fetchUser,
  };
}
