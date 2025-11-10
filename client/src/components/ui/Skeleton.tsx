import React from 'react';

export const Skeleton = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div 
    className={`animate-pulse bg-gray-200 rounded-md ${className}`} 
    {...props} 
  />
);

export const SkeletonText = ({ lines = 1, className = '' }: { lines?: number; className?: string }) => (
  <div className={`space-y-2 ${className}`}>
    {Array.from({ length: lines }).map((_, i) => (
      <Skeleton key={i} className="h-4 w-full" />
    ))}
  </div>
);

export const StatCardSkeleton = () => (
  <div className="bg-white p-6 rounded-lg shadow">
    <Skeleton className="h-5 w-32 mb-2" />
    <Skeleton className="h-8 w-24 mb-2" />
    <Skeleton className="h-4 w-40" />
  </div>
);

export const ChartSkeleton = () => (
  <div className="bg-white p-6 rounded-lg shadow h-80">
    <div className="flex justify-between items-center mb-6">
      <Skeleton className="h-6 w-40" />
      <Skeleton className="h-4 w-32" />
    </div>
    <div className="h-full w-full flex items-end space-x-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="flex-1 flex flex-col items-center">
          <Skeleton className="w-full mb-2" style={{ height: `${Math.random() * 80 + 20}%` }} />
          <Skeleton className="h-4 w-8" />
        </div>
      ))}
    </div>
  </div>
);

export const ActivityItemSkeleton = () => (
  <div className="p-4 hover:bg-gray-50 transition-colors">
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <Skeleton className="h-10 w-10 rounded-full" />
        <div>
          <Skeleton className="h-4 w-32 mb-2" />
          <Skeleton className="h-3 w-24" />
        </div>
      </div>
      <Skeleton className="h-4 w-20" />
    </div>
  </div>
);

export const EmptyState = ({
  title,
  description,
  icon: Icon,
  action,
  className = ''
}: {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  action?: React.ReactNode;
  className?: string;
}) => (
  <div className={`text-center py-12 ${className}`}>
    <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-gray-100 mb-4">
      <Icon className="h-6 w-6 text-gray-400" />
    </div>
    <h3 className="mt-2 text-sm font-medium text-gray-900">{title}</h3>
    <p className="mt-1 text-sm text-gray-500">{description}</p>
    {action && <div className="mt-6">{action}</div>}
  </div>
);
