import WebSocket from 'ws';

// Test WebSocket connection
async function testWebSocket() {
  console.log('Testing WebSocket connection...');
  
  // Get a test token (you'll need to replace this with a real token)
  const testToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJpYXQiOjE2MzU0NzIwMDAsImV4cCI6MTYzNTQ3NTYwMH0.test-signature';
  
  const wsUrl = `ws://localhost:5001/api/v2/dashboard/ws?token=${testToken}`;
  
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
  });
  
  ws.on('message', (data) => {
    const message = JSON.parse(data.toString());
    console.log('📨 Received message:', message);
  });
  
  ws.on('close', (code, reason) => {
    console.log(`🔌 WebSocket closed: ${code} - ${reason}`);
  });
  
  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error.message);
  });
  
  // Close after 5 seconds
  setTimeout(() => {
    ws.close();
    process.exit(0);
  }, 5000);
}

testWebSocket().catch(console.error);
