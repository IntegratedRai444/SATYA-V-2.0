declare global {
  namespace Express {
    // Custom user type
    interface User {
      id: string;
      email: string;
      email_verified: boolean;
      role?: string;
      user_metadata?: Record<string, unknown>;
      app_metadata?: Record<string, unknown>;
      phone_verified?: boolean;
    }

    // Extend the Request interface
    interface Request {
      id?: string;
      user?: User;
      // Add other custom properties here
    }
  }
}

export {};
