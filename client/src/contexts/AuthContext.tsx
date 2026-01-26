import React, { createContext, ReactNode } from 'react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';

interface AuthContextType {
  user: ReturnType<typeof useSupabaseAuth>['user'];
  session: ReturnType<typeof useSupabaseAuth>['session'];
  loading: ReturnType<typeof useSupabaseAuth>['loading'];
  error: ReturnType<typeof useSupabaseAuth>['error'];
  signIn: ReturnType<typeof useSupabaseAuth>['signIn'];
  signUp: ReturnType<typeof useSupabaseAuth>['signUp'];
  signOut: ReturnType<typeof useSupabaseAuth>['signOut'];
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export { AuthContext };

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const auth = useSupabaseAuth();
  
  const value: AuthContextType = {
    ...auth,
    isAuthenticated: !!auth.user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;
