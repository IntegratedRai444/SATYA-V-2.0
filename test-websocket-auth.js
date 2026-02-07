import WebSocket from 'ws';

// Test WebSocket connection with real authentication flow
async function testWebSocketAuth() {
  console.log('Testing WebSocket authentication...');
  
  try {
    // First, get a real token by registering/logging in
    const registerResponse = await fetch('http://localhost:5001/api/v2/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: 'test@example.com',
        password: 'testpassword123',
        username: 'testuser'
      })
    });
    
    if (!registerResponse.ok) {
      console.log('User might already exist, trying login...');
      // Try login instead
      const loginResponse = await fetch('http://localhost:5001/api/v2/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'test@example.com',
          password: 'testpassword123'
        })
      });
      
      if (!loginResponse.ok) {
        throw new Error('Login failed');
      }
      
      const loginData = await loginResponse.json();
      const token = loginData.data?.session?.access_token || loginData.access_token;
      
      if (!token) {
        throw new Error('No token received');
      }
      
      console.log('✅ Got authentication token');
      await testWebSocketConnection(token);
    } else {
      const registerData = await registerResponse.json();
      const token = registerData.data?.session?.access_token || registerData.access_token;
      
      if (!token) {
        throw new Error('No token received');
      }
      
      console.log('✅ Registered and got authentication token');
      await testWebSocketConnection(token);
    }
  } catch (error) {
    console.error('❌ Authentication failed:', error.message);
  }
}

async function testWebSocketConnection(token) {
  console.log('Testing WebSocket connection with real token...');
  
  const wsUrl = `ws://localhost:5001/api/v2/dashboard/ws?token=${token}`;
  
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

testWebSocketAuth().catch(console.error);
