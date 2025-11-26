import { useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';

interface PageTransitionProps {
  children: React.ReactNode;
  className?: string;
}

export const PageTransition = ({ children, className }: PageTransitionProps) => {
  const location = useLocation();
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 300);
    return () => clearTimeout(timer);
  }, [location.pathname]);

  return (
    <div
      className={`transition-all duration-300 ease-in-out ${
        isAnimating ? 'opacity-0 translate-y-5' : 'opacity-100 translate-y-0'
      } ${className || ''}`}
    >
      {children}
    </div>
  );
};

export default PageTransition;
