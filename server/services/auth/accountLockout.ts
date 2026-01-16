import { dbManager } from '../../db';
import { users } from '@shared/schema';
import { eq } from 'drizzle-orm';
import { logger } from '../../config/logger';

const MAX_FAILED_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes

export async function handleFailedLogin(email: string, ipAddress: string): Promise<{ isLocked: boolean; remainingAttempts: number }> {
  try {
    const user = (await dbManager.find('users', { email }, { limit: 1 }))[0];

    if (!user) {
      // Don't reveal if user exists
      return { isLocked: false, remainingAttempts: 0 };
    }

    const now = new Date();
    const nowISO = now.toISOString();
    const lastFailedLogin = user.last_failed_login ? new Date(user.last_failed_login) : null;
    const lockoutExpires = lastFailedLogin ? new Date(lastFailedLogin.getTime() + LOCKOUT_DURATION_MS) : null;
    
    // Reset failed attempts if lockout period has passed
    const failedAttempts = (lockoutExpires && lockoutExpires > now) 
      ? (user.failed_login_attempts || 0) + 1 
      : 1;

    const isLocked = failedAttempts >= MAX_FAILED_ATTEMPTS;
    const remainingAttempts = Math.max(0, MAX_FAILED_ATTEMPTS - failedAttempts);

    await dbManager.update('users', user.id.toString(), {
      failed_login_attempts: failedAttempts,
      last_failed_login: nowISO,
      is_locked: isLocked,
      lockout_until: isLocked ? new Date(now.getTime() + LOCKOUT_DURATION_MS).toISOString() : null,
      updated_at: nowISO
    });

    if (isLocked) {
      logger.warn(`Account locked for user ${email} from IP ${ipAddress}`, {
        userId: user.id,
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

export async function resetFailedLoginAttempts(userId: number): Promise<void> {
  try {
    const nowISO = new Date().toISOString();
    await dbManager.update('users', userId.toString(), {
      failed_login_attempts: 0,
      last_failed_login: null,
      is_locked: false,
      lockout_until: null,
      updated_at: nowISO
    });
  } catch (error) {
    logger.error('Failed to reset failed login attempts:', error);
  }
}

export async function isAccountLocked(email: string): Promise<{ isLocked: boolean; remainingTimeMs?: number }> {
  try {
    const user = (await dbManager.find('users', { email }, { limit: 1 }))[0];

    if (!user || !user.last_failed_login) {
      return { isLocked: false };
    }

    const lastFailedLogin = new Date(user.last_failed_login);
    const lockoutExpires = new Date(lastFailedLogin.getTime() + LOCKOUT_DURATION_MS);
    const now = new Date();

    if (user.failed_login_attempts >= MAX_FAILED_ATTEMPTS && lockoutExpires > now) {
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
