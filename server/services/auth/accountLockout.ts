import { supabase } from '../../config/supabase';
import { logger } from '../../config/logger';

const MAX_FAILED_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes

export async function handleFailedLogin(email: string, ipAddress: string): Promise<{ isLocked: boolean; remainingAttempts: number }> {
  try {
    const { data: user, error } = await supabase
      .from('users')
      .select('*')
      .eq('email', email)
      .limit(1);

    if (error || !user || user.length === 0) {
      // Don't reveal if user exists
      return { isLocked: false, remainingAttempts: 0 };
    }

    const now = new Date();
    const nowISO = now.toISOString();
    const userData = user[0]; // Get the first user from the array
    const lastFailedLogin = userData?.last_failed_login ? new Date(userData.last_failed_login) : null;
    const lockoutExpires = lastFailedLogin ? new Date(lastFailedLogin.getTime() + LOCKOUT_DURATION_MS) : null;
    
    // Reset failed attempts if lockout period has passed
    const failedAttempts = (lockoutExpires && lockoutExpires > now) 
      ? (userData?.failed_login_attempts || 0) + 1 
      : 1;

    const isLocked = failedAttempts >= MAX_FAILED_ATTEMPTS;
    const remainingAttempts = Math.max(0, MAX_FAILED_ATTEMPTS - failedAttempts);

    await supabase
      .from('users')
      .update({
        failed_login_attempts: failedAttempts,
        last_failed_login: nowISO,
        is_locked: isLocked,
        lockout_until: isLocked ? new Date(now.getTime() + LOCKOUT_DURATION_MS).toISOString() : null,
        updated_at: nowISO
      })
      .eq('id', userData.id);

    if (isLocked) {
      logger.warn(`Account locked for user ${email} from IP ${ipAddress}`, {
        userId: userData.id,
        ipAddress,
        timestamp: now.toISOString()
      });
    }

    return { isLocked, remainingAttempts };
  } catch (error) {
    logger.error('Failed to handle failed login attempt:', error);
    return { isLocked: false, remainingAttempts: 0 }; // Fail open to prevent denial of service
  }
}

export async function resetFailedLoginAttempts(userId: string): Promise<void> {
  try {
    const nowISO = new Date().toISOString();
    await supabase
      .from('users')
      .update({
        failed_login_attempts: 0,
        last_failed_login: null,
        is_locked: false,
        lockout_until: null,
        updated_at: nowISO
      })
      .eq('id', userId);
  } catch (error) {
    logger.error('Failed to reset failed login attempts:', error);
  }
}

export async function isAccountLocked(email: string): Promise<{ isLocked: boolean; remainingTimeMs?: number }> {
  try {
    const { data: user, error } = await supabase
      .from('users')
      .select('*')
      .eq('email', email)
      .limit(1);

    if (error || !user || user.length === 0 || !user[0].last_failed_login) {
      return { isLocked: false };
    }

    const userData = user[0];
    const lastFailedLogin = new Date(userData.last_failed_login);
    const lockoutExpires = new Date(lastFailedLogin.getTime() + LOCKOUT_DURATION_MS);
    const now = new Date();

    if (userData.failed_login_attempts >= MAX_FAILED_ATTEMPTS && lockoutExpires > now) {
      return { 
        isLocked: true, 
        remainingTimeMs: lockoutExpires.getTime() - now.getTime() 
      };
    }

    return { isLocked: false };
  } catch (error) {
    logger.error('Failed to check account lock status:', error);
    return { isLocked: false }; // Fail open
  }
}
