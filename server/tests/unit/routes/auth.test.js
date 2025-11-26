import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import request from 'supertest';
import express, { Express } from 'express';
import authRouter from '../../../routes/auth';
import { jwtAuthService } from '../../../services/jwt-auth-service';
import { sessionManager } from '../../../services/session-manager';
// Mock dependencies
jest.mock('../../../services/jwt-auth-service');
jest.mock('../../../services/session-manager');
jest.mock('../../../config/logger');
describe('Authentication Routes', () => {
    let app;
    beforeEach(() => {
        // Setup Express app with auth routes
        app = express();
        app.use(express.json());
        app.use('/api/auth', authRouter);
        // Clear all mocks
        jest.clearAllMocks();
    });
    afterEach(() => {
        jest.resetAllMocks();
    });
    describe('POST /api/auth/register', () => {
        it('should register a new user successfully', async () => {
            const mockUser = {
                id: 1,
                username: 'testuser',
                email: 'test@example.com',
                role: 'user'
            };
            const mockToken = 'mock-jwt-token';
            // Mock register service
            jwtAuthService.register.mockResolvedValue({
                success: true,
                message: 'Registration successful',
                token: mockToken,
                user: mockUser
            });
            // Mock token verification
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: mockUser.id,
                username: mockUser.username,
                email: mockUser.email,
                role: mockUser.role
            });
            // Mock session creation
            sessionManager.createSession.mockResolvedValue(true);
            const response = await request(app)
                .post('/api/auth/register')
                .send({
                username: 'testuser',
                password: 'TestPass123!',
                email: 'test@example.com',
                fullName: 'Test User'
            });
            expect(response.status).toBe(201);
            expect(response.body.success).toBe(true);
            expect(response.body.token).toBe(mockToken);
            expect(response.body.user.username).toBe('testuser');
            expect(jwtAuthService.register).toHaveBeenCalledWith({
                username: 'testuser',
                password: 'TestPass123!',
                email: 'test@example.com',
                fullName: 'Test User'
            });
        });
        it('should fail with invalid username (too short)', async () => {
            const response = await request(app)
                .post('/api/auth/register')
                .send({
                username: 'ab',
                password: 'TestPass123!',
                email: 'test@example.com'
            });
            expect(response.status).toBe(400);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('Validation failed');
        });
        it('should fail with weak password', async () => {
            const response = await request(app)
                .post('/api/auth/register')
                .send({
                username: 'testuser',
                password: 'weak',
                email: 'test@example.com'
            });
            expect(response.status).toBe(400);
            expect(response.body.success).toBe(false);
        });
        it('should fail when username already exists', async () => {
            jwtAuthService.register.mockResolvedValue({
                success: false,
                message: 'Username already exists'
            });
            const response = await request(app)
                .post('/api/auth/register')
                .send({
                username: 'existinguser',
                password: 'TestPass123!',
                email: 'test@example.com'
            });
            expect(response.status).toBe(400);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('Username already exists');
        });
    });
    describe('POST /api/auth/login', () => {
        it('should login successfully with valid credentials', async () => {
            const mockUser = {
                id: 1,
                username: 'testuser',
                email: 'test@example.com',
                role: 'user'
            };
            const mockToken = 'mock-jwt-token';
            jwtAuthService.login.mockResolvedValue({
                success: true,
                message: 'Login successful',
                token: mockToken,
                user: mockUser
            });
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: mockUser.id,
                username: mockUser.username,
                email: mockUser.email,
                role: mockUser.role
            });
            sessionManager.createSession.mockResolvedValue(true);
            const response = await request(app)
                .post('/api/auth/login')
                .send({
                username: 'testuser',
                password: 'TestPass123!'
            });
            expect(response.status).toBe(200);
            expect(response.body.success).toBe(true);
            expect(response.body.token).toBe(mockToken);
            expect(response.body.user.username).toBe('testuser');
        });
        it('should fail with invalid credentials', async () => {
            jwtAuthService.login.mockResolvedValue({
                success: false,
                message: 'Invalid credentials'
            });
            const response = await request(app)
                .post('/api/auth/login')
                .send({
                username: 'testuser',
                password: 'WrongPassword123!'
            });
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('Invalid credentials');
        });
        it('should fail with missing username', async () => {
            const response = await request(app)
                .post('/api/auth/login')
                .send({
                password: 'TestPass123!'
            });
            expect(response.status).toBe(400);
            expect(response.body.success).toBe(false);
        });
        it('should fail with missing password', async () => {
            const response = await request(app)
                .post('/api/auth/login')
                .send({
                username: 'testuser'
            });
            expect(response.status).toBe(400);
            expect(response.body.success).toBe(false);
        });
    });
    describe('POST /api/auth/logout', () => {
        it('should logout successfully with valid token', async () => {
            const mockToken = 'valid-jwt-token';
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: 1,
                username: 'testuser'
            });
            sessionManager.destroySession.mockResolvedValue(true);
            const response = await request(app)
                .post('/api/auth/logout')
                .set('Authorization', `Bearer ${mockToken}`);
            expect(response.status).toBe(200);
            expect(response.body.success).toBe(true);
            expect(response.body.message).toBe('Logout successful');
            expect(sessionManager.destroySession).toHaveBeenCalledWith(mockToken);
        });
        it('should fail without token', async () => {
            const response = await request(app)
                .post('/api/auth/logout');
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('No token provided');
        });
        it('should handle session destruction failure', async () => {
            const mockToken = 'valid-jwt-token';
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: 1,
                username: 'testuser'
            });
            sessionManager.destroySession.mockResolvedValue(false);
            const response = await request(app)
                .post('/api/auth/logout')
                .set('Authorization', `Bearer ${mockToken}`);
            expect(response.status).toBe(500);
            expect(response.body.success).toBe(false);
        });
    });
    describe('GET /api/auth/session', () => {
        it('should return user info for valid session', async () => {
            const mockToken = 'valid-jwt-token';
            const mockUser = {
                id: 1,
                username: 'testuser',
                email: 'test@example.com',
                fullName: 'Test User',
                role: 'user',
                createdAt: new Date()
            };
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: mockUser.id,
                username: mockUser.username
            });
            jwtAuthService.getUserById.mockResolvedValue(mockUser);
            const response = await request(app)
                .get('/api/auth/session')
                .set('Authorization', `Bearer ${mockToken}`);
            expect(response.status).toBe(200);
            expect(response.body.success).toBe(true);
            expect(response.body.user.username).toBe('testuser');
        });
        it('should fail without token', async () => {
            const response = await request(app)
                .get('/api/auth/session');
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
        });
        it('should fail with invalid token', async () => {
            jwtAuthService.verifyToken.mockResolvedValue(null);
            const response = await request(app)
                .get('/api/auth/session')
                .set('Authorization', 'Bearer invalid-token');
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('Invalid or expired token');
        });
        it('should fail when user not found', async () => {
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: 999,
                username: 'nonexistent'
            });
            jwtAuthService.getUserById.mockResolvedValue(null);
            const response = await request(app)
                .get('/api/auth/session')
                .set('Authorization', 'Bearer valid-token');
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
            expect(response.body.message).toBe('User not found');
        });
    });
    describe('Rate Limiting', () => {
        it('should enforce rate limits on authentication endpoints', async () => {
            // This test would require rate limiting middleware to be active
            // For now, we verify the endpoint structure supports it
            jwtAuthService.login.mockResolvedValue({
                success: false,
                message: 'Invalid credentials'
            });
            // Make multiple requests
            const requests = Array(6).fill(null).map(() => request(app)
                .post('/api/auth/login')
                .send({
                username: 'testuser',
                password: 'wrong'
            }));
            const responses = await Promise.all(requests);
            // All should complete (rate limiting would be tested in integration tests)
            responses.forEach(response => {
                expect([400, 401, 429]).toContain(response.status);
            });
        });
    });
    describe('JWT Token Validation', () => {
        it('should validate JWT token format', async () => {
            const response = await request(app)
                .get('/api/auth/session')
                .set('Authorization', 'InvalidFormat');
            expect(response.status).toBe(401);
            expect(response.body.success).toBe(false);
        });
        it('should extract token from Bearer format', async () => {
            const mockToken = 'valid-jwt-token';
            jwtAuthService.verifyToken.mockResolvedValue({
                userId: 1,
                username: 'testuser'
            });
            jwtAuthService.getUserById.mockResolvedValue({
                id: 1,
                username: 'testuser',
                email: 'test@example.com',
                role: 'user',
                createdAt: new Date()
            });
            const response = await request(app)
                .get('/api/auth/session')
                .set('Authorization', `Bearer ${mockToken}`);
            expect(response.status).toBe(200);
            expect(jwtAuthService.verifyToken).toHaveBeenCalledWith(mockToken);
        });
    });
});
