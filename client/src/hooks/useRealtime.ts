import { useContext } from 'react';
import { RealtimeContext, RealtimeContextType } from '../contexts/RealtimeContext';

export const useRealtime = (): RealtimeContextType => {
  const context = useContext(RealtimeContext);
  if (!context) {
    throw new Error('useRealtime must be used within a RealtimeProvider');
  }
  return context;
};
