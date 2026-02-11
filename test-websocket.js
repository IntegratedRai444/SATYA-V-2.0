import WebSocket from 'ws';

// Test WebSocket connection and message handling
const ws = new WebSocket('ws://localhost:5001/api/v2/dashboard/ws');

ws.on('open', () => {
  console.log('✅ WebSocket connected successfully');
  
  // Test subscription to a job
  ws.send(JSON.stringify({
    type: 'SUBSCRIBE_JOB',
    payload: { jobId: 'test-job-123' },
    timestamp: new Date().toISOString(),
    id: Math.random().toString(36).substr(2, 9)
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);
  console.log('📨 Received message:', message);
  
  // Check if message types match what we expect
  if (['JOB_PROGRESS', 'JOB_COMPLETED', 'JOB_FAILED'].includes(message.type)) {
    console.log('✅ Correct message type received:', message.type);
    if (message.payload && message.payload.jobId === 'test-job-123') {
      console.log('✅ Correct job ID match');
    }
  } else {
    console.log('❌ Unexpected message type:', message.type);
  }
});

ws.on('error', (error) => {
  console.error('❌ WebSocket error:', error);
});

ws.on('close', () => {
  console.log('🔌 WebSocket connection closed');
});

console.log('🔍 Testing WebSocket message types...');
