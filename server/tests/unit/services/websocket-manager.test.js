import { Server } from 'http';
import { WebSocket } from 'ws';
import { webSocketManager } from '../../../../server/services/websocket-manager';
import { createServer } from 'http';
import { AddressInfo } from 'net';
import { logger } from '../../../../server/config';
describe('WebSocketManager', () => {
    let server;
    let port;
    const testUserId = 123;
    const testUsername = 'testuser';
    beforeAll((done) => {
        server = createServer();
        webSocketManager.initialize(server);
        server.listen(0, () => {
            port = server.address().port;
            done();
        });
    });
    afterAll((done) => {
        webSocketManager.destroy();
        server.close(done);
    });
    afterEach(() => {
        jest.clearAllMocks();
    });
    describe('Connection Management', () => {
        it('should accept valid WebSocket connections', (done) => {
            const ws = new WebSocket(`ws://localhost:${port}/ws?token=test-token`);
            ws.on('open', () => {
                expect(ws.readyState).toBe(WebSocket.OPEN);
                ws.close();
                done();
            });
            ws.on('error', (error) => {
                done.fail(`WebSocket connection failed: ${error.message}`);
            });
        });
        it('should reject connections without token', (done) => {
            const ws = new WebSocket(`ws://localhost:${port}/ws`);
            ws.on('error', (error) => {
                expect(error.message).toContain('401');
                done();
            });
        });
    });
    describe('Message Handling', () => {
        let ws;
        beforeEach((done) => {
            ws = new WebSocket(`ws://localhost:${port}/ws?token=test-token`);
            ws.on('open', () => done());
        });
        afterEach(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        });
        it('should handle ping messages', (done) => {
            ws.on('message', (data) => {
                const message = JSON.parse(data.toString());
                if (message.type === 'pong') {
                    done();
                }
            });
            ws.send(JSON.stringify({ type: 'ping' }));
        });
        it('should handle job subscription', (done) => {
            const testJobId = 'test-job-123';
            ws.on('message', (data) => {
                const message = JSON.parse(data.toString());
                if (message.type === 'job_status' && message.data.jobId === testJobId) {
                    expect(message.data.status).toBeDefined();
                    done();
                }
            });
            ws.send(JSON.stringify({
                type: 'subscribe_job',
                jobId: testJobId
            }));
        });
    });
    describe('Error Handling', () => {
        it('should handle invalid message formats', (done) => {
            const ws = new WebSocket(`ws://localhost:${port}/ws?token=test-token`);
            ws.on('open', () => {
                ws.send('invalid-json');
                ws.on('message', (data) => {
                    const message = JSON.parse(data.toString());
                    if (message.type === 'error') {
                        expect(message.error).toContain('Invalid message format');
                        ws.close();
                        done();
                    }
                });
            });
        });
    });
});
