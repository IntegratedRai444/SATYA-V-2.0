import WebSocket from 'ws';

// Simple WebSocket test with a mock token for testing connection only
async function testWebSocketConnection() {
  console.log('Testing WebSocket connection...');
  
  // Create a test JWT token (this would normally come from Supabase)
  // For testing purposes, we'll create a simple token structure
  const testPayload = {
    sub: 'test-user-id',
    email: 'test@example.com',
    aud: 'authenticated',
    role: 'authenticated',
    exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour expiry
    iat: Math.floor(Date.now() / 1000)
  };
  
  // Create a mock JWT (just for testing the WebSocket connection)
  const header = { alg: 'HS256', typ: 'JWT' };
  const encodedHeader = Buffer.from(JSON.stringify(header)).toString('base64url');
  const encodedPayload = Buffer.from(JSON.stringify(testPayload)).toString('base64url');
  const signature = 'mock-signature'; // This would normally be HMAC-SHA256
  
  const mockToken = `${encodedHeader}.${encodedPayload}.${signature}`;
  
  console.log('Generated mock token for testing...');
  
  const wsUrl = `ws://localhost:5001/api/v2/dashboard/ws?token=${mockToken}`;
  
  const ws = new WebSocket(wsUrl);
  
  ws.on('open', () => {
    console.log('✅ WebSocket connected successfully!');
    
    // Test SUBSCRIBE_JOB message
    ws.send(JSON.stringify({
      type: 'SUBSCRIBE_JOB',
      payload: { jobId: 'test-job-123' },
      timestamp: Date.now(),
      id: 'test-message-1'
    }));
    
    // Test ping
    setTimeout(() => {
      ws.send(JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      }));
    }, 1000);
    
    // Test unsubscribe
    setTimeout(() => {
      ws.send(JSON.stringify({
        type: 'UNSUBSCRIBE_JOB',
        payload: { jobId: 'test-job-123' },
        timestamp: Date.now(),
        id: 'test-message-2'
      }));
    }, 2000);
  });
  
  ws.on('message', (data) => {
    const message = JSON.parse(data.toString());
    console.log('📨 Received message:', message);
  });
  
  ws.on('close', (code, reason) => {
    console.log(`🔌 WebSocket closed: ${code} - ${reason}`);
    process.exit(0);
  });
  
  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error.message);
    process.exit(1);
  });
  
  // Close after 5 seconds
  setTimeout(() => {
    ws.close();
  }, 5000);
}

testWebSocketConnection().catch(console.error);
