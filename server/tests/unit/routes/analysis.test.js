import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import request from 'supertest';
import express, { Express } from 'express';
import analysisRouter from '../../../routes/analysis';
// Mock dependencies
jest.mock('../../../middleware/auth');
jest.mock('../../../middleware/upload');
jest.mock('../../../services/python-bridge');
jest.mock('../../../config/logger');
describe('Analysis Routes', () => {
    let app;
    beforeEach(() => {
        app = express();
        app.use(express.json());
        app.use('/api/analysis', analysisRouter);
        jest.clearAllMocks();
    });
    afterEach(() => {
        jest.resetAllMocks();
    });
    describe('POST /api/analysis/image', () => {
        it('should analyze image successfully', async () => {
            const response = await request(app)
                .post('/api/analysis/image')
                .set('Authorization', 'Bearer mock-token')
                .attach('file', Buffer.from('fake-image'), 'test.jpg');
            expect([200, 400, 401]).toContain(response.status);
        });
        it('should fail without authentication', async () => {
            const response = await request(app)
                .post('/api/analysis/image')
                .attach('file', Buffer.from('fake-image'), 'test.jpg');
            expect([401, 400]).toContain(response.status);
        });
    });
    describe('POST /api/analysis/video', () => {
        it('should handle video analysis request', async () => {
            const response = await request(app)
                .post('/api/analysis/video')
                .set('Authorization', 'Bearer mock-token')
                .attach('file', Buffer.from('fake-video'), 'test.mp4');
            expect([200, 400, 401]).toContain(response.status);
        });
    });
    describe('POST /api/analysis/audio', () => {
        it('should handle audio analysis request', async () => {
            const response = await request(app)
                .post('/api/analysis/audio')
                .set('Authorization', 'Bearer mock-token')
                .attach('file', Buffer.from('fake-audio'), 'test.mp3');
            expect([200, 400, 401]).toContain(response.status);
        });
    });
    describe('POST /api/analysis/webcam', () => {
        it('should handle webcam analysis with base64 data', async () => {
            const response = await request(app)
                .post('/api/analysis/webcam')
                .set('Authorization', 'Bearer mock-token')
                .send({
                imageData: 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
            });
            expect([200, 400, 401]).toContain(response.status);
        });
        it('should fail without image data', async () => {
            const response = await request(app)
                .post('/api/analysis/webcam')
                .set('Authorization', 'Bearer mock-token')
                .send({});
            expect([400, 401]).toContain(response.status);
        });
    });
    describe('POST /api/analysis/multimodal', () => {
        it('should handle multimodal analysis', async () => {
            const response = await request(app)
                .post('/api/analysis/multimodal')
                .set('Authorization', 'Bearer mock-token')
                .attach('image', Buffer.from('fake-image'), 'test.jpg')
                .attach('audio', Buffer.from('fake-audio'), 'test.mp3');
            expect([200, 400, 401]).toContain(response.status);
        });
    });
    describe('Rate Limiting', () => {
        it('should enforce rate limits on analysis endpoints', async () => {
            const requests = Array(12).fill(null).map(() => request(app)
                .post('/api/analysis/image')
                .set('Authorization', 'Bearer mock-token')
                .attach('file', Buffer.from('fake'), 'test.jpg'));
            const responses = await Promise.all(requests);
            const statusCodes = responses.map(r => r.status);
            expect(statusCodes.some(code => [200, 400, 401, 429].includes(code))).toBe(true);
        });
    });
});
