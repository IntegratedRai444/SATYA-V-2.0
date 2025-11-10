import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

type AppState = {
  theme: 'light' | 'dark' | 'system';
  isOnline: boolean;
  isKeyboardUser: boolean;
  notifications: Notification[];
  notificationSettings: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
  userPreferences: {
    language: string;
    timezone: string;
    dateFormat: string;
    timeFormat: '12h' | '24h';
  };
};

type Action =
  | { type: 'SET_THEME'; payload: 'light' | 'dark' | 'system' }
  | { type: 'SET_ONLINE_STATUS'; payload: boolean }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'UPDATE_NOTIFICATION_SETTINGS'; payload: Partial<AppState['notificationSettings']> }
  | { type: 'UPDATE_USER_PREFERENCES'; payload: Partial<AppState['userPreferences']> };

type Notification = {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
};

const initialState: AppState = {
  theme: (localStorage.getItem('theme') as AppState['theme']) || 'system',
  isOnline: navigator.onLine,
  isKeyboardUser: false,
  notifications: [],
  notificationSettings: {
    email: localStorage.getItem('notification_email') !== 'false',
    push: localStorage.getItem('notification_push') !== 'false',
    sound: localStorage.getItem('notification_sound') !== 'false',
  },
  userPreferences: {
    language: localStorage.getItem('language') || 'en-US',
    timezone: localStorage.getItem('timezone') || Intl.DateTimeFormat().resolvedOptions().timeZone,
    dateFormat: localStorage.getItem('dateFormat') || 'MM/dd/yyyy',
    timeFormat: (localStorage.getItem('timeFormat') as '12h' | '24h') || '12h',
  },
};

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
} | null>(null);

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_THEME':
      localStorage.setItem('theme', action.payload);
      return { ...state, theme: action.payload };
    case 'SET_ONLINE_STATUS':
      return { ...state, isOnline: action.payload };
    case 'ADD_NOTIFICATION':
      return { ...state, notifications: [...state.notifications, action.payload] };
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload),
      };
    case 'UPDATE_NOTIFICATION_SETTINGS':
      const newNotificationSettings = { ...state.notificationSettings, ...action.payload };
      Object.entries(action.payload).forEach(([key, value]) => {
        localStorage.setItem(`notification_${key}`, String(value));
      });
      return { ...state, notificationSettings: newNotificationSettings };
    case 'UPDATE_USER_PREFERENCES':
      const newUserPreferences = { ...state.userPreferences, ...action.payload };
      Object.entries(action.payload).forEach(([key, value]) => {
        localStorage.setItem(key, String(value));
      });
      return { ...state, userPreferences: newUserPreferences };
    default:
      return state;
  }
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5 * 60 * 1000, // 5 minutes
        cacheTime: 30 * 60 * 1000, // 30 minutes
        refetchOnWindowFocus: false,
        retry: 1,
      },
    },
  });

  // Handle online/offline status
  useEffect(() => {
    const handleOnline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: true });
    const handleOffline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: false });

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Detect keyboard users for better accessibility
    const handleFirstTab = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        document.body.classList.add('keyboard-user');
        dispatch({ type: 'SET_KEYBOARD_USER', payload: true });
        window.removeEventListener('keydown', handleFirstTab);
      }
    };

    window.addEventListener('keydown', handleFirstTab);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('keydown', handleFirstTab);
    };
  }, []);

  // Apply theme and preferences
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    
    if (state.theme === 'system') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.add(isDark ? 'dark' : 'light');
    } else {
      root.classList.add(state.theme);
    }
  }, [state.theme, state.userPreferences]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      <QueryClientProvider client={queryClient}>
        {children}
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}
