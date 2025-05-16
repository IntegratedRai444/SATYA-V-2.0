import { useState, useEffect } from "react";
import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "next-themes";
import Dashboard from "@/pages/Dashboard";
import Scan from "@/pages/Scan";
import History from "@/pages/History";
import Settings from "@/pages/Settings";
import Help from "@/pages/Help";
import NotFound from "@/pages/not-found";
import Layout from "@/components/layout/Layout";
import AuthGuard from "@/components/auth/AuthGuard";
import LoginForm from "@/components/auth/LoginForm";

// Define which routes require authentication
const protectedRoutes = ['/history', '/settings', '/scan'];

function Router() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={Dashboard} />
        <Route path="/login">
          {() => (
            <div className="container mx-auto py-8 px-4">
              <div className="max-w-md mx-auto">
                <LoginForm />
              </div>
            </div>
          )}
        </Route>
        
        {/* Protected Routes */}
        <Route path="/scan">
          {() => (
            <AuthGuard>
              <Scan />
            </AuthGuard>
          )}
        </Route>
        
        <Route path="/history">
          {() => (
            <AuthGuard>
              <History />
            </AuthGuard>
          )}
        </Route>
        
        <Route path="/settings">
          {() => (
            <AuthGuard>
              <Settings />
            </AuthGuard>
          )}
        </Route>
        
        {/* Public Routes */}
        <Route path="/help" component={Help} />
        <Route component={NotFound} />
      </Switch>
    </Layout>
  );
}

function App() {
  const [serverInitialized, setServerInitialized] = useState(false);
  
  // Initialize Python server when app loads
  useEffect(() => {
    // We'll connect to the Python backend when needed
    // This will be handled by the auth components
    setServerInitialized(true);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
