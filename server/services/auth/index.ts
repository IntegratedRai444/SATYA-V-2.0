import jwt from 'jsonwebtoken';
import { logger } from '../../config/logger';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const JWT_ALGORITHM = 'HS256';

export interface TokenPayload {
  userId: string;
  email: string;
  role?: string;
  [key: string]: any;
}

export const generateToken = (payload: TokenPayload, expiresIn: string | number = '1h'): string => {
  const options = {
    algorithm: JWT_ALGORITHM as jwt.Algorithm,
    expiresIn
  } as jwt.SignOptions;
  
  return jwt.sign(
    { ...payload, iat: Math.floor(Date.now() / 1000) } as object,
    JWT_SECRET,
    options
  );
};

export const verifyToken = async (token: string): Promise<TokenPayload | null> => {
  try {
    const decoded = jwt.verify(token, JWT_SECRET, { algorithms: [JWT_ALGORITHM] });
    return decoded as TokenPayload;
  } catch (error) {
    logger.error('Token verification failed:', error);
    return null;
  }
};

export const decodeToken = (token: string): TokenPayload | null => {
  try {
    const decoded = jwt.decode(token) as TokenPayload | null;
    return decoded;
  } catch (error) {
    logger.error('Token decoding failed:', error);
    return null;
  }
};

export const refreshToken = (token: string, expiresIn: string = '7d'): string | null => {
  const payload = decodeToken(token);
  if (!payload) return null;
  
  // Remove iat and exp from payload
  const { iat, exp, ...cleanPayload } = payload;
  
  return generateToken(cleanPayload, expiresIn);
};

export default {
  generateToken,
  verifyToken,
  decodeToken,
  refreshToken
};
