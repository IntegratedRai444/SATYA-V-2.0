import React, { useState } from 'react';
import { useLocation } from 'wouter';
<<<<<<< HEAD
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { useToast } from "../../hooks/use-toast";
import { LockKeyhole, Shield, User } from "lucide-react";
=======
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { LockKeyhole, Shield, User } from "lucide-react";

// Import the Python bridge for authentication
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
import { login } from '../../lib/auth';

interface LoginFormProps {
  onLoginSuccess?: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
<<<<<<< HEAD
  const [error, setError] = useState<string | null>(null);
=======
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
<<<<<<< HEAD
    setError(null);
    if (!username || !password) {
      setError("Please enter both username and password");
=======
    
    if (!username || !password) {
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter both username and password"
      });
      return;
    }
<<<<<<< HEAD
    setIsLoading(true);
    try {
      const result = await login(username, password);
      console.log('Login result:', result);
      if (result.success) {
        localStorage.setItem('satyaai_token', result.token);
        localStorage.setItem('satyaai_user', JSON.stringify(result.user));
=======
    
    setIsLoading(true);
    
    try {
      const result = await login(username, password);
      
      if (result.success) {
        // Store token in localStorage
        localStorage.setItem('satyaai_token', result.token);
        localStorage.setItem('satyaai_user', JSON.stringify(result.user));
        
        // Success toast
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
        toast({
          title: "Login Successful",
          description: `Welcome back, ${result.user.username}!`,
        });
<<<<<<< HEAD
=======
        
        // Call success callback or redirect
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
        if (onLoginSuccess) {
          onLoginSuccess();
        } else {
          setLocation('/dashboard');
        }
      } else {
<<<<<<< HEAD
        setError(result.message || "Invalid username or password");
=======
        // Show error message
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
        toast({
          variant: "destructive",
          title: "Login Failed",
          description: result.message || "Invalid username or password"
        });
      }
    } catch (error) {
<<<<<<< HEAD
      setError("An unexpected error occurred. Please try again.");
      console.error('Login error:', error);
=======
      console.error('Login error:', error);
      
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
      toast({
        variant: "destructive",
        title: "Login Error",
        description: "An unexpected error occurred. Please try again."
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
<<<<<<< HEAD
    <Card className="w-full max-w-md mx-auto border border-primary/20 relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      )}
=======
    <Card className="w-full max-w-md mx-auto border border-primary/20">
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold text-center flex items-center justify-center">
          <Shield className="h-6 w-6 mr-2 text-primary" />
          SatyaAI Authentication
        </CardTitle>
        <CardDescription className="text-center">
          Enter your credentials to access SatyaAI deepfake detection
        </CardDescription>
      </CardHeader>
<<<<<<< HEAD
      {error && (
        <div className="px-6 pb-2 text-red-600 text-center font-medium">{error}</div>
      )}
=======
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <div className="relative">
              <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                id="username"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="pl-10"
                disabled={isLoading}
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <LockKeyhole className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-10"
                disabled={isLoading}
              />
            </div>
          </div>
          <div className="text-sm text-muted-foreground">
            <p>For demo purposes, any non-empty username and password will work.</p>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            type="submit"
            className="w-full"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Logging in...
              </>
            ) : (
              'Sign In'
            )}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
};

export default LoginForm;