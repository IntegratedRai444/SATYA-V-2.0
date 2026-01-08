import { verify } from 'jsonwebtoken';
import { Request } from 'express';
import { JWT_SECRET } from '../config/constants';

interface TokenPayload {
  userId: string;
  sessionId: string;
  [key: string]: any;
}

export const extractTokenFromQuery = (req: Request): string | null => {
  const token = req.query.token as string;
  return token || null;
};

export const verifyToken = (token: string): TokenPayload | null => {
  try {
    const decoded = verify(token, JWT_SECRET) as TokenPayload;
    return decoded;
  } catch (error) {
    return null;
  }
};

export const getClientIp = (req: Request): string => {
  return (
    (req.headers['x-forwarded-for'] as string)?.split(',')[0] ||
    req.socket.remoteAddress ||
    'unknown-ip'
  );
};
