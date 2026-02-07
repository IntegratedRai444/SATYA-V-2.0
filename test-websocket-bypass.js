import WebSocket from 'ws';

// Test WebSocket connection without authentication to verify server is working
async function testWebSocketConnection() {
  console.log('Testing WebSocket connection (bypassing auth)...');
  
  // Connect directly to the WebSocket server path
  const wsUrl = `ws://localhost:5001/api/v2/dashboard/ws`;
  
  const ws = new WebSocket(wsUrl);
  
  ws.on('open', () => {
    console.log('✅ WebSocket connected successfully!');
    
    // Test ping
    ws.send(JSON.stringify({
      type: 'ping',
      timestamp: Date.now()
    }));
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
  
  // Close after 3 seconds
  setTimeout(() => {
    ws.close();
  }, 3000);
}

testWebSocketConnection().catch(console.error);
