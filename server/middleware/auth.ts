export type AuthUser = {
  id: string;
  email: string;
  role: string;
  email_verified: boolean;
  user_metadata?: Record<string, unknown>;
};
