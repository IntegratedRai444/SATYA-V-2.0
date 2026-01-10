import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Form, FormField, FormItem, FormLabel, FormMessage } from '../../components/ui/form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// Simple spinner icon component
const Spinner = ({ className }: { className?: string }) => (
  <svg
    className={cn("animate-spin h-5 w-5", className)}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

// Helper function to combine class names
function cn(...classes: (string | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

type FormValues = {
  email: string;
  password: string;
  confirmPassword: string;
};

// Move password validation functions here since we're having path resolution issues
interface PasswordValidationResult {
  isValid: boolean;
  errors: string[];
}

const validatePassword = (password: string): PasswordValidationResult => {
  const errors: string[] = [];
  
  if (password.length < 8) {
    errors.push("Password must be at least 8 characters long");
  }
  
  if (!/[A-Z]/.test(password)) {
    errors.push("Password must contain at least one uppercase letter");
  }
  
  if (!/[a-z]/.test(password)) {
    errors.push("Password must contain at least one lowercase letter");
  }
  
  if (!/[0-9]/.test(password)) {
    errors.push("Password must contain at least one number");
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

const getPasswordErrorMessage = (errors: string[]): string => {
  if (errors.length === 0) return '';
  return errors.join("\n");
};

// Password validation schema
const passwordSchema = z.string()
  .min(8, 'Password must be at least 8 characters')
  .refine((val) => /[A-Z]/.test(val), 'Must contain at least one uppercase letter')
  .refine((val) => /[a-z]/.test(val), 'Must contain at least one lowercase letter')
  .refine((val) => /[0-9]/.test(val), 'Must contain at least one number');

// Form validation schema
const formSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: passwordSchema,
  confirmPassword: z.string()
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export function RegisterForm() {
  const { register, error: authError, clearError } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: '',
      password: '',
      confirmPassword: '',
    },
  });
  
  const password = form.watch('password');

  // Clear any existing errors when form is mounted
  useEffect(() => {
    clearError();
  }, [clearError]);

  // Update local error state when authError changes
  useEffect(() => {
    if (authError) {
      setError(authError);
    }
  }, [authError]);

  const onSubmit = async (values: FormValues) => {
    try {
      setIsLoading(true);
      setError('');
      
      // Additional password validation
      const passwordValidation = validatePassword(values.password);
      if (!passwordValidation.isValid) {
        setError(getPasswordErrorMessage(passwordValidation.errors));
        return;
      }
      
      // Generate a username from email (or use a better method if needed)
      const username = values.email.split('@')[0];
      
      // Call the register function from AuthContext
      const success = await register(username, values.email, values.password);
      
      if (success) {
        // On successful registration and auto-login, redirect to dashboard
        navigate('/dashboard', { replace: true });
      }
    } catch (err) {
      console.error('Registration error:', err);
      setError(err instanceof Error ? err.message : 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold">Create an account</CardTitle>
        <CardDescription>
          Enter your email and password to create your account
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            {error && (
              <div className="p-3 text-sm text-red-600 bg-red-50 rounded-md">
                {error}
              </div>
            )}
            
            <FormField
              control={form.control as any}
              name="email"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email</FormLabel>
                  <Input
                    type="email"
                    placeholder="name@example.com"
                    autoComplete="email"
                    {...field}
                  />
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control as any}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <div className="flex items-center justify-between">
                    <FormLabel>Password</FormLabel>
                    <span className="text-xs text-muted-foreground">
                      Min 8 characters
                    </span>
                  </div>
                  <Input
                    type="password"
                    placeholder="••••••••"
                    autoComplete="new-password"
                    {...field}
                  />
                  <FormMessage />
                  <div className="mt-1 text-xs text-muted-foreground">
                    Must include:
                    <ul className="mt-1 space-y-0.5">
                      <li className={/[A-Z]/.test(password || '') ? 'text-green-500' : ''}>
                        • At least one uppercase letter
                      </li>
                      <li className={/[a-z]/.test(password || '') ? 'text-green-500' : ''}>
                        • At least one lowercase letter
                      </li>
                      <li className={/[0-9]/.test(password || '') ? 'text-green-500' : ''}>
                        • At least one number
                      </li>
                    </ul>
                  </div>
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control as any}
              name="confirmPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Confirm Password</FormLabel>
                  <Input
                    type="password"
                    placeholder="••••••••"
                    autoComplete="new-password"
                    {...field}
                  />
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading && (
                <Spinner className="w-4 h-4 mr-2" />
              )}
              Create Account
            </Button>
            
            <p className="px-8 text-sm text-center text-muted-foreground">
              Already have an account?{' '}
              <a
                href="/login"
                className="underline hover:text-primary"
                onClick={(e) => {
                  e.preventDefault();
                  navigate('/login');
                }}
              >
                Sign in
              </a>
            </p>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
