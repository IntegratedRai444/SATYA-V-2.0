import React, { useState, useEffect } from 'react';
import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "./components/ui/toaster";
import { TooltipProvider } from "./components/ui/tooltip";
import { ThemeProvider } from "next-themes";
import Dashboard from "./pages/Dashboard";
import Scan from "./pages/Scan";
import History from "./pages/History";
import Settings from "./pages/Settings";
import Help from "./pages/Help";
import NotFound from "./pages/not-found";
import Layout from "./components/layout/Layout";
import AuthGuard from "./components/auth/AuthGuard";
import LoginForm from "./components/auth/LoginForm";

function LoadingPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background">
      <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-6"></div>
      <h2 className="text-2xl font-bold mb-2">Loading SatyaAI...</h2>
      <p className="text-muted-foreground">Checking system health and configuration...</p>
    </div>
  );
}

function Router() {
  return (
    <Switch>
      {/* Public Routes */}
      <Route path="/login">
        {() => (
          <AuthGuard>
            <div className="container mx-auto py-8 px-4">
              <div className="max-w-md mx-auto">
                <LoginForm />
              </div>
            </div>
          </AuthGuard>
        )}
      </Route>
      <Route path="/help" component={Help} />
      {/* Protected Routes */}
      <Route path="/">
        {() => (
          <AuthGuard>
            <Layout>
              <Switch>
                <Route path="/" component={Dashboard} />
                <Route path="/scan" component={Scan} />
                <Route path="/history" component={History} />
                <Route path="/history/:id" component={History} />
                <Route path="/settings" component={Settings} />
                <Route component={NotFound} />
              </Switch>
            </Layout>
          </AuthGuard>
        )}
      </Route>
    </Switch>
  );
}

function App() {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [, setLocation] = useLocation();

  useEffect(() => {
    async function fetchHealthAndConfig() {
      try {
        const healthRes = await fetch("/api/health");
        const configRes = await fetch("/api/config");
        const healthData = await healthRes.json();
        const configData = await configRes.json();
        setHealth(healthData);
        setConfig(configData);
      } catch (error) {
        setHealth(null);
        setConfig(null);
      } finally {
        setLoading(false);
      }
    }
    fetchHealthAndConfig();
  }, []);

  if (loading) {
    return <LoadingPage />;
  }

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
