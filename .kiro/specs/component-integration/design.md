# Design Document - Component Integration

## Overview

This document outlines the technical design for integrating all currently unused frontend components into the SatyaAI application. The integration will ensure all components are actively used while maintaining code quality and user experience.

## Architecture

### Integration Strategy

The integration follows a **bottom-up approach**:
1. **Foundation Layer**: Integrate contexts and providers
2. **Service Layer**: Ensure all services are used by hooks
3. **Hook Layer**: Integrate hooks into components
4. **Component Layer**: Integrate components into pages
5. **Page Layer**: Verify all pages use appropriate components

### File Modification Order

1. `main.tsx` - Add missing context providers
2. Layout components - Integrate Sidebar, Navbar, Footer
3. Page components - Add missing component imports
4. Component files - Ensure proper exports

## Components and Interfaces

### 1. Dashboard Page Integration

**File**: `client/src/pages/Dashboard.tsx`

**Components to Add**:
- `RecentActivity` - Display recent scans in a sidebar or bottom section
- `AnalysisProgress` - Show ongoing analysis progress
- `ProgressTracker` - Track multiple analysis tasks
- `AnalysisResults` - Display completed analysis results

**Layout Structure**:
```tsx
<Dashboard>
  <DashboardHeader />
  <DashboardStats />
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div className="lg:col-span-2">
      <AnalysisProgress />
      <ProgressTracker />
      <AnalysisResults />
    </div>
    <div className="lg:col-span-1">
      <RecentActivity />
    </div>
  </div>
</Dashboard>
```

### 2. Detection Tools Page Integration

**File**: `client/src/pages/DetectionTools.tsx`

**Components to Add**:
- `DetectionToolsGrid` - Main grid container
- `DetectionToolCard` - Individual tool cards

**Implementation**:
```tsx
import { DetectionToolsGrid } from '@/components/detection/DetectionToolsGrid';
import { DetectionToolCard } from '@/components/detection/DetectionToolCard';

const tools = [
  { id: 'image', name: 'Image Analysis', icon: '🖼️', accuracy: 94.7, route: '/image-analysis' },
  { id: 'video', name: 'Video Analysis', icon: '🎥', accuracy: 89.3, route: '/video-analysis' },
  { id: 'audio', name: 'Audio Analysis', icon: '🎵', accuracy: 91.7, route: '/audio-analysis' },
  { id: 'webcam', name: 'Live Webcam', icon: '📹', accuracy: 92.1, route: '/webcam-live' },
];

<DetectionToolsGrid>
  {tools.map(tool => (
    <DetectionToolCard key={tool.id} {...tool} />
  ))}
</DetectionToolsGrid>
```

### 3. Landing Page Integration

**File**: `client/src/pages/LandingPage.tsx`

**Components to Add**:
- `HeroSection` - Main hero with CTA
- `AuthenticityScore` - Display authenticity metrics
- `ParticleBackground` - Animated background
- `DetectionToolsPreview` - Preview of detection tools
- `CircularProgress` - For score visualization
- `AuthenticityScoreCard` - Score display card

**Layout Structure**:
```tsx
<LandingPage>
  <ParticleBackground />
  <HeroSection>
    <AuthenticityScoreCard>
      <CircularProgress value={97} />
      <AuthenticityScore score={97} />
    </AuthenticityScoreCard>
  </HeroSection>
  <DetectionToolsPreview />
</LandingPage>
```

### 4. Home Page Integration

**File**: `client/src/pages/Home.tsx`

**Components to Add**:
- `AuthenticityScoreCard` - Display user's average score
- `CircularProgress` - Visual score representation
- `ParticleBackground` - Background effects

### 5. MainLayout Integration

**File**: `client/src/components/layout/MainLayout.tsx`

**Components to Add**:
- `Sidebar` - Left navigation sidebar
- `Navbar` - Top navigation bar
- `Footer` - Bottom footer
- `NotificationBell` - Notification icon in navbar

**Current vs New Structure**:
```tsx
// Current (simplified)
<MainLayout>
  <Header />
  <main>{children}</main>
</MainLayout>

// New (integrated)
<MainLayout>
  <Navbar>
    <NotificationBell />
  </Navbar>
  <div className="flex">
    <Sidebar />
    <main className="flex-1">
      {children}
    </main>
  </div>
  <Footer />
</MainLayout>
```

### 6. Upload Analysis Page Integration

**File**: `client/src/pages/UploadAnalysis.tsx`

**Components to Add**:
- `BatchUploader` - Multi-file upload component

**Implementation**:
```tsx
import { BatchUploader } from '@/components/batch/BatchUploader';

<UploadAnalysis>
  <BatchUploader
    onUpload={handleBatchUpload}
    maxFiles={10}
    acceptedTypes={['image/*', 'video/*', 'audio/*']}
  />
</UploadAnalysis>
```

### 7. Audio Analysis Page Integration

**File**: `client/src/pages/AudioAnalysis.tsx`

**Components to Add**:
- `AudioAnalyzer` - Real-time audio visualization

**Implementation**:
```tsx
import { AudioAnalyzer } from '@/components/realtime/AudioAnalyzer';

<AudioAnalysis>
  {isLiveMode && (
    <AudioAnalyzer
      audioStream={audioStream}
      onAnalysisUpdate={handleUpdate}
    />
  )}
</AudioAnalysis>
```

### 8. Scan Page Integration

**File**: `client/src/pages/Scan.tsx`

**Components to Add**:
- `ScanProgress` - Progress indicator for scans

**Implementation**:
```tsx
import { ScanProgress } from '@/components/scans/ScanProgress';

<Scan>
  {isScanning && (
    <ScanProgress
      progress={scanProgress}
      status={scanStatus}
      estimatedTime={estimatedTime}
    />
  )}
</Scan>
```

### 9. Help Page Integration

**File**: `client/src/pages/Help.tsx`

**Components to Add**:
- `ChatInterface` - AI assistant chat
- `WelcomeMessage` - Initial greeting
- `ChatMessage` - Message display

**Implementation**:
```tsx
import { ChatInterface } from '@/components/chat/ChatInterface';
import { WelcomeMessage } from '@/components/chat/WelcomeMessage';

<Help>
  <ChatInterface>
    <WelcomeMessage />
    {/* Chat messages will be rendered here */}
  </ChatInterface>
</Help>
```

## Data Models

### Component Props Interfaces

```typescript
// DetectionToolCard
interface DetectionToolCardProps {
  id: string;
  name: string;
  icon: string;
  description?: string;
  accuracy: number;
  route: string;
  features?: string[];
}

// RecentActivity
interface RecentActivityProps {
  limit?: number;
  onItemClick?: (scanId: string) => void;
}

// AnalysisProgress
interface AnalysisProgressProps {
  analyses: Analysis[];
  onCancel?: (analysisId: string) => void;
}

// BatchUploader
interface BatchUploaderProps {
  onUpload: (files: File[]) => Promise<void>;
  maxFiles?: number;
  acceptedTypes?: string[];
  maxFileSize?: number;
}

// AudioAnalyzer
interface AudioAnalyzerProps {
  audioStream: MediaStream;
  onAnalysisUpdate?: (data: AudioAnalysisData) => void;
  visualizationType?: 'waveform' | 'spectrogram';
}

// ScanProgress
interface ScanProgressProps {
  progress: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  estimatedTime?: number;
  fileName?: string;
}

// ChatInterface
interface ChatInterfaceProps {
  onSendMessage?: (message: string) => void;
  messages?: ChatMessage[];
  isLoading?: boolean;
}
```

## Context Integration

### Main.tsx Updates

```tsx
import { AppProvider } from './contexts/AppContext';
import { BatchProcessingProvider } from './contexts/BatchProcessingContext';
import { RealtimeProvider } from './contexts/RealtimeContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

root.render(
  <StrictMode>
    <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <AuthProvider>
            <AppProvider>
              <WebSocketProvider>
                <RealtimeProvider>
                  <BatchProcessingProvider>
                    <RouterProvider router={router} />
                  </BatchProcessingProvider>
                </RealtimeProvider>
              </WebSocketProvider>
            </AppProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </HelmetProvider>
  </StrictMode>
);
```

## Hook Integration Strategy

### Hook Usage Mapping

| Hook | Used In | Purpose |
|------|---------|---------|
| `useDashboard` | Dashboard.tsx | Fetch dashboard data |
| `useDashboardStats` | Dashboard.tsx | Get statistics |
| `useDashboardWebSocket` | Dashboard.tsx | Real-time updates |
| `useAnalysis` | Analysis pages | Manage analysis state |
| `useAnalytics` | Analytics.tsx | Fetch analytics data |
| `useApi` | All service hooks | API communication |
| `useBatchProcessing` | UploadAnalysis.tsx | Batch upload management |
| `useDetections` | Detection pages | Detection operations |
| `useNavigation` | Layout components | Navigation helpers |
| `useNotifications` | NotificationBell | Notification management |
| `useSettings` | Settings.tsx | User settings |
| `useUser` | All pages | User data access |
| `useWebSocket` | Real-time components | WebSocket connections |
| `useScanWebSocket` | Scan.tsx | Scan progress updates |

## Service Integration

### Service Layer Architecture

```
Components/Pages
      ↓
   Hooks (Custom)
      ↓
   Services
      ↓
   API Client
      ↓
   Backend API
```

### Service Usage

```typescript
// In hooks/useDashboard.ts
import { dashboardService } from '@/services/dashboardService';
import { useApi } from './useApi';

export const useDashboard = () => {
  const { request } = useApi();
  
  const fetchDashboardData = async () => {
    return request(() => dashboardService.getDashboardData());
  };
  
  return { fetchDashboardData };
};
```

## Error Handling

### Component-Level Error Boundaries

Each major page section will be wrapped with ErrorBoundary:

```tsx
<ErrorBoundary fallback={<ErrorFallback />}>
  <RecentActivity />
</ErrorBoundary>
```

### Loading States

All async components will show loading states:

```tsx
{isLoading ? <LoadingSpinner /> : <RecentActivity data={data} />}
```

## Testing Strategy

### Integration Testing Approach

1. **Component Rendering**: Verify each component renders without errors
2. **Props Validation**: Ensure props are passed correctly
3. **Hook Integration**: Test hooks return expected data
4. **Service Calls**: Mock service calls and verify responses
5. **User Interactions**: Test click handlers and form submissions

### Testing Priority

1. **High Priority**: Dashboard, DetectionTools, MainLayout
2. **Medium Priority**: Landing page, Analysis pages
3. **Low Priority**: Help/Chat, Settings

## Performance Considerations

### Lazy Loading

Components will be lazy-loaded where appropriate:

```typescript
const RecentActivity = lazy(() => import('@/components/analysis/RecentActivity'));
const BatchUploader = lazy(() => import('@/components/batch/BatchUploader'));
```

### Memoization

Heavy components will use React.memo:

```typescript
export const DetectionToolCard = React.memo(({ ...props }) => {
  // Component implementation
});
```

### Code Splitting

Large features (chat, batch upload) will be code-split:

```typescript
const ChatInterface = lazy(() => 
  import(/* webpackChunkName: "chat" */ '@/components/chat/ChatInterface')
);
```

## Deployment Strategy

### Phased Rollout

1. **Phase 1**: Layout components (Sidebar, Navbar, Footer)
2. **Phase 2**: Dashboard components
3. **Phase 3**: Detection and analysis components
4. **Phase 4**: Advanced features (chat, batch, realtime)

### Rollback Plan

- Each phase will be a separate commit
- Feature flags for new components
- Easy rollback to previous version if issues arise

## Success Metrics

1. **All 127 files actively used** in the application
2. **No console errors** on any page
3. **All pages render correctly** with new components
4. **Performance maintained** (no significant slowdown)
5. **User experience improved** with new features

---

*Design Version: 1.0*
*Last Updated: 2025-01-10*
