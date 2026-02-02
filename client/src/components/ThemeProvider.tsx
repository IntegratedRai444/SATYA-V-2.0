import React from 'react';

// Simple Theme Provider to replace next-themes
export const SimpleThemeProvider = ({ children }: { children: React.ReactNode }) => {
  React.useEffect(() => {
    // Force dark theme
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);
  
  return <>{children}</>;
};
