# Frontend File Usage Analysis

## Summary

**Total Frontend Files:** 127 files
- .tsx files: 70
- .ts files: 47
- .js files: 7
- .css files: 2
- .html files: 1

## Files Actually Used in Application

### ✅ PAGES (16 files - ALL USED via Router)
All page components are loaded via the router with lazy loading:

1. ✅ **Home.tsx** - /home route
2. ✅ **Dashboard.tsx** - / and /dashboard routes (DEFAULT)
3. ✅ **Login.tsx** - /login route
4. ✅ **Analytics.tsx** - /analytics route
5. ✅ **DetectionTools.tsx** - /detection-tools route
6. ✅ **UploadAnalysis.tsx** - /upload route
7. ✅ **Scan.tsx** - /scan/:id route
8. ✅ **ImageAnalysis.tsx** - /image-analysis route
9. ✅ **VideoAnalysis.tsx** - /video-analysis route
10. ✅ **AudioAnalysis.tsx** - /audio-analysis route
11. ✅ **History.tsx** - /history and /scan-history routes
12. ✅ **Settings.tsx** - /settings route
13. ✅ **Help.tsx** - /help route
14. ✅ **WebcamLive.tsx** - /webcam and /webcam-live routes
15. ✅ **LandingPage.tsx** - / route (public)
16. ✅ **NotFound.tsx** - /404 route

### ✅ LAYOUT COMPONENTS (Used in Router)
1. ✅ **MainLayout.tsx** - Main app layout wrapper
2. ⚠️ **Navbar.tsx** - Likely used in MainLayout
3. ⚠️ **Sidebar.tsx** - Likely used in MainLayout
4. ⚠️ **Header.tsx** - Possibly used
5. ⚠️ **Footer.tsx** - Possibly used
6. ❓ **ModernNavbar.tsx** - Duplicate? Check if used

### ✅ UI COMPONENTS (Used in Router & Pages)
1. ✅ **LoadingState.tsx** - Used in router
2. ✅ **ErrorBoundary.tsx** - Used in router
3. ✅ **PageTransition.tsx** - Used in router
4. ✅ **Toaster.tsx** - Used in router
5. ⚠️ **button.tsx, card.tsx, input.tsx, etc.** - Likely used in pages

### ✅ CONTEXTS (Used in main.tsx)
1. ✅ **AuthContext.tsx** - Used in main.tsx
2. ⚠️ **AppContext.tsx** - Check usage
3. ⚠️ **BatchProcessingContext.tsx** - Check usage
4. ⚠️ **RealtimeContext.tsx** - Check usage
5. ⚠️ **WebSocketContext.tsx** - Check usage

### ✅ UTILS
1. ✅ **router.tsx** - Main router, used in main.tsx

## Potentially Unused Files

### ❌ COMPONENTS - Analysis
- **components/analysis/** (4 files) - May be used in Dashboard/Analytics pages
- **components/auth/AuthGuard.tsx** - Replaced by ProtectedRoute in router?
- **components/batch/BatchUploader.tsx** - Check if used in any page
- **components/chat/** (3 files) - Check if chat feature is implemented
- **components/dashboard/index.ts** - Check exports
- **components/detection/** (2 files) - Likely used in DetectionTools page
- **components/home/** (4 files) - Likely used in Home/LandingPage
- **components/landing/** (3 files) - Likely used in LandingPage
- **components/notifications/NotificationBell.tsx** - Check if used in layout
- **components/realtime/AudioAnalyzer.tsx** - Check if used
- **components/scans/ScanProgress.tsx** - Likely used in Scan page

### ❌ HOOKS - Need to Check Usage in Pages
All 18 hooks need to be checked if they're imported in pages/components:
- useAnalysis, useAnalytics, useApi, useBatchProcessing
- useDashboard, useDashboardStats, useDashboardWebSocket
- useDetections, useLocalStorage, useNavigation
- useNotifications, useScanWebSocket, useSettings
- useUser, useWebSocket, use-toast, use-media-query, use-mobile

### ❌ SERVICES - Need to Check Usage
All 7 services need verification:
- analytics.ts, api.ts, auth.ts
- chatService.ts, dashboardService.ts
- userService.ts, websocket.ts

### ❌ LIB - Need to Check Usage
All 10 lib files need verification:
- api-client.ts, api.ts, auth.ts, config.ts
- file-utils.ts, fileUpload.ts, queryClient.ts
- reactQuery.ts, types.ts, utils.ts

## Recommendations

### 1. Immediate Actions
- **Remove duplicate navbars:** You have both Navbar.tsx and ModernNavbar.tsx
- **Check AuthGuard.tsx:** Seems replaced by ProtectedRoute in router
- **Verify chat components:** If chat feature isn't implemented, remove these

### 2. Deep Analysis Needed
Run a more detailed import analysis on:
- All hooks (check which pages use them)
- All services (check which hooks/pages use them)
- All lib utilities (check usage across codebase)
- Component subdirectories (analysis, detection, home, landing, etc.)

### 3. Likely Safe to Keep
- All UI components (button, card, input, etc.) - standard UI library
- All page components - all are routed
- MainLayout, Sidebar, Navbar - core layout
- AuthContext - used in main.tsx
- router.tsx - core routing

### 4. Files to Investigate
Priority files to check if they're actually used:
1. components/auth/AuthGuard.tsx
2. components/layout/ModernNavbar.tsx (vs Navbar.tsx)
3. components/chat/* (3 files)
4. components/batch/BatchUploader.tsx
5. components/realtime/AudioAnalyzer.tsx

## Next Steps

1. **Read each page file** to see what components/hooks they import
2. **Check MainLayout.tsx** to see which layout components it uses
3. **Verify service usage** in hooks and pages
4. **Create a dependency graph** to visualize actual usage
5. **Remove confirmed unused files** after verification

---

*Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")*
