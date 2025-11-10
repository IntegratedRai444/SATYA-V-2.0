# MainLayout Explained 🎨

## What is MainLayout?

**MainLayout** is a **wrapper component** that provides a consistent layout structure for most pages in your SatyaAI application. Think of it as a "frame" or "template" that wraps around your page content.

---

## 📍 Location
**File:** `client/src/components/layout/MainLayout.tsx`

---

## 🎯 Purpose

MainLayout provides:
1. **Consistent Navigation** - Same navbar and sidebar on every page
2. **Unified Look** - Same layout structure across the app
3. **Code Reusability** - Write layout code once, use everywhere
4. **Easy Maintenance** - Update navigation in one place

---

## 🏗️ Structure

### Visual Layout:
```
┌─────────────────────────────────────────────────────────┐
│  NAVBAR (Fixed Top - 64px height)                      │
│  [Logo] [Nav Items] [User Menu] [Notifications]        │
└─────────────────────────────────────────────────────────┘
┌──────────────┬──────────────────────────────────────────┐
│              │                                          │
│  SIDEBAR     │  MAIN CONTENT AREA                      │
│  (Fixed)     │  (Scrollable)                           │
│  280px wide  │                                          │
│              │  ┌────────────────────────────────┐     │
│  Navigation: │  │                                │     │
│  - Dashboard │  │  <Outlet />                    │     │
│  - Tools     │  │  (Your page content goes here) │     │
│  - Upload    │  │                                │     │
│  - Scans     │  │                                │     │
│  - History   │  │                                │     │
│  - Analytics │  │                                │     │
│  - Settings  │  │                                │     │
│  - Help      │  └────────────────────────────────┘     │
│              │                                          │
│              │  ┌────────────────────────────────┐     │
│              │  │  FOOTER                        │     │
│              │  │  [Copyright] [Links] [Social]  │     │
│              │  └────────────────────────────────┘     │
└──────────────┴──────────────────────────────────────────┘
```

---

## 🧩 Components Inside MainLayout

### 1. **Navbar** (Top Bar)
**File:** `client/src/components/layout/Navbar.tsx`
**Contains:**
- Logo (SatyaAI)
- Navigation items (Home, Scan, History)
- User menu (Profile, Settings, Logout)
- NotificationBell component

### 2. **Sidebar** (Left Navigation)
**File:** `client/src/components/layout/Sidebar.tsx`
**Contains:**
- Dashboard link
- Detection Tools section
  - Image Analysis
  - Video Analysis
  - Audio Analysis
  - Webcam Live
- Management section
  - Upload & Analyze
  - Scan History
  - Analytics
  - Settings
  - Help & Support

### 3. **Footer** (Bottom)
**File:** `client/src/components/layout/Footer.tsx`
**Contains:**
- Copyright info
- Links (Terms, Privacy, Contact)
- Social media links

### 4. **Outlet** (Content Area)
**What is it?** React Router component that renders the current page
**Example:** When you visit `/analytics`, the Analytics page renders here

### 5. **Toaster** (Notifications)
**What is it?** Toast notification system for alerts and messages

---

## 🔄 How MainLayout Works

### The Code Breakdown:

```typescript
const MainLayout = () => {
  const location = useLocation();

  // Check if current page is auth page or landing
  const isAuthPage = ['/login', '/register', '/forgot-password', '/reset-password', '/'].includes(location.pathname);

  // If auth page, show simple layout (no navbar/sidebar)
  if (isAuthPage) {
    return (
      <div className="min-h-screen bg-bg-primary">
        <Outlet />  {/* Just the page content */}
        <Toaster />
      </div>
    );
  }

  // Otherwise, show full layout with navbar, sidebar, footer
  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      <Navbar />  {/* Top navigation */}
      
      <div className="flex pt-16">
        <Sidebar />  {/* Left navigation */}
        
        <main className="flex-1 ml-[280px]">
          <div className="p-8">
            <Outlet />  {/* Your page content here */}
          </div>
          <Footer />  {/* Bottom footer */}
        </main>
      </div>
      
      <Toaster />  {/* Toast notifications */}
    </div>
  );
};
```

---

## 📄 Which Pages Use MainLayout?

### ✅ Pages WITH MainLayout (13 pages)
These pages are wrapped in the full layout with Navbar, Sidebar, and Footer:

1. **Analytics** (`/analytics`)
2. **DetectionTools** (`/detection-tools`)
3. **ImageAnalysis** (`/image-analysis`)
4. **VideoAnalysis** (`/video-analysis`)
5. **AudioAnalysis** (`/audio-analysis`)
6. **WebcamLive** (`/webcam-live`)
7. **UploadAnalysis** (`/upload`)
8. **Scan** (`/scan/:id`)
9. **History** (`/history`)
10. **Settings** (`/settings`)
11. **Help** (`/help`)
12. **Home** (`/home`)
13. **NotFound** (`/404`)

### ❌ Pages WITHOUT MainLayout (3 pages)
These pages have their own custom layout:

1. **Dashboard** (`/` and `/dashboard`) - Standalone layout
2. **LandingPage** (`/` public) - Custom landing layout
3. **Login** (`/login`) - Simple auth layout

---

## 🎨 Layout Comparison

### With MainLayout:
```
┌─────────────────────────────────────┐
│  Navbar                             │ ← Always visible
├──────────┬──────────────────────────┤
│ Sidebar  │  Your Page Content       │
│          │  (Analytics, Settings,   │ ← Your page renders here
│          │   Help, etc.)            │
│          ├──────────────────────────┤
│          │  Footer                  │ ← Always visible
└──────────┴──────────────────────────┘
```

### Without MainLayout (Dashboard):
```
┌─────────────────────────────────────┐
│                                     │
│  Your Complete Custom Layout        │
│  (Dashboard has its own design)     │ ← Full control
│                                     │
│                                     │
└─────────────────────────────────────┘
```

---

## 🔧 Technical Details

### Styling:
- **Background:** Dark theme (`bg-[#0a0a0a]`)
- **Navbar Height:** 64px (4rem)
- **Sidebar Width:** 280px
- **Content Padding:** 32px (p-8)
- **Content Margin:** 280px left (ml-[280px]) to account for sidebar

### Responsive Behavior:
- Navbar is fixed at top
- Sidebar is fixed on left
- Main content scrolls independently
- Footer is at bottom of content area

### Router Integration:
- Uses `<Outlet />` from React Router
- Checks current route with `useLocation()`
- Conditionally shows/hides layout based on route

---

## 🎯 Why Two Layout Types?

### MainLayout (Type 2)
**Used for:** Regular app pages
**Why:** Consistent navigation, easy to find features
**Example:** Settings, Analytics, Help

### Standalone (Type 1)
**Used for:** Special pages
**Why:** Custom design, unique experience
**Example:** Dashboard (main hub), Landing page (marketing)

---

## 🚀 How to Use MainLayout

### In Router (`client/src/utils/router.tsx`):

```typescript
// Pages WITH MainLayout
{
  element: <MainLayout />,  // Wrap with MainLayout
  children: [
    {
      path: '/analytics',
      element: <Analytics />  // This renders inside <Outlet />
    },
    {
      path: '/settings',
      element: <Settings />  // This also renders inside <Outlet />
    }
  ]
}

// Pages WITHOUT MainLayout
{
  path: '/dashboard',
  element: <Dashboard />  // Renders directly, no wrapper
}
```

---

## 📊 MainLayout Benefits

### ✅ Advantages:
1. **Consistency** - Same navigation everywhere
2. **Efficiency** - Write layout code once
3. **Maintainability** - Update in one place
4. **User Experience** - Familiar navigation
5. **Accessibility** - Consistent keyboard navigation

### ⚠️ When NOT to Use:
1. Landing pages (need custom design)
2. Auth pages (simple, focused)
3. Dashboard (unique hero section)
4. Special marketing pages

---

## 🎨 Visual Example

### Page WITH MainLayout (Analytics):
```
User visits: /analytics

Router renders:
  <MainLayout>
    <Navbar />
    <Sidebar />
    <main>
      <Analytics />  ← Your page here
    </main>
    <Footer />
  </MainLayout>

User sees:
  - Navbar at top
  - Sidebar on left
  - Analytics content in center
  - Footer at bottom
```

### Page WITHOUT MainLayout (Dashboard):
```
User visits: /dashboard

Router renders:
  <Dashboard />  ← Just the page, no wrapper

User sees:
  - Custom dashboard layout
  - No navbar/sidebar
  - Full control of design
```

---

## 🔍 Key Takeaways

1. **MainLayout = Wrapper** that adds Navbar + Sidebar + Footer
2. **Used by 13 pages** for consistent navigation
3. **NOT used by 3 pages** (Dashboard, Landing, Login) for custom designs
4. **Contains 5 components**: Navbar, Sidebar, Footer, Outlet, Toaster
5. **Responsive** with fixed navbar and sidebar
6. **Smart routing** - hides layout on auth pages

---

## 🎯 In Simple Terms

**MainLayout is like a picture frame:**
- The frame (Navbar, Sidebar, Footer) stays the same
- The picture (your page content) changes
- Some special pictures (Dashboard, Landing) don't need a frame

**Your Dashboard doesn't use MainLayout because:**
- It has a custom hero section
- It has its own unique design
- It's the main landing page
- It needs full creative control

**Other pages use MainLayout because:**
- They need consistent navigation
- They're utility pages (Settings, Help, Analytics)
- They benefit from the sidebar menu
- They don't need custom layouts

---

**File:** `client/src/components/layout/MainLayout.tsx`  
**Type:** Layout Wrapper Component  
**Used By:** 13 pages  
**Contains:** Navbar, Sidebar, Footer, Outlet, Toaster
