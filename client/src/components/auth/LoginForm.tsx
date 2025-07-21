import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { useToast } from "../../hooks/use-toast";
import { LockKeyhole, Shield, User } from "lucide-react";
import { login } from '../../lib/auth';

interface LoginFormProps {
  onLoginSuccess?: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!username || !password) {
      setError("Please enter both username and password");
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter both username and password"
      });
      return;
    }
    setIsLoading(true);
    try {
      const result = await login(username, password);
      console.log('Login result:', result);
      if (result.success) {
        localStorage.setItem('satyaai_token', result.token);
        localStorage.setItem('satyaai_user', JSON.stringify(result.user));
        toast({
          title: "Login Successful",
          description: `Welcome back, ${result.user.username}!`,
        });
        if (onLoginSuccess) {
          onLoginSuccess();
        } else {
          setLocation('/dashboard');
        }
      } else {
        setError(result.message || "Invalid username or password");
        toast({
          variant: "destructive",
          title: "Login Failed",
          description: result.message || "Invalid username or password"
        });
      }
    } catch (error) {
      setError("An unexpected error occurred. Please try again.");
      console.error('Login error:', error);
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
    <Card className="w-full max-w-md mx-auto border border-primary/20 relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold text-center flex items-center justify-center">
          <Shield className="h-6 w-6 mr-2 text-primary" />
          SatyaAI Authentication
        </CardTitle>
        <CardDescription className="text-center">
          Enter your credentials to access SatyaAI deepfake detection
        </CardDescription>
      </CardHeader>
      {error && (
        <div className="px-6 pb-2 text-red-600 text-center font-medium">{error}</div>
      )}
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