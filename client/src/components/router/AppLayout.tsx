import React from 'react';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import { Toaster } from '@/components/ui/toaster';
import MainLayout from '@/components/layout/MainLayout';
import ScrollToTop from '@/components/common/ScrollToTop';

const AppLayout: React.FC = () => {
  return (
    <ErrorBoundary level="app">
      <ScrollToTop />
      <MainLayout />
      <Toaster />
    </ErrorBoundary>
  );
};

export default AppLayout;