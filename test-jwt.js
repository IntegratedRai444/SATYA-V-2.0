// Test JWT verification
import jwt from 'jsonwebtoken';

// Sample JWT from frontend logs (truncated)
const sampleToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ0YnBiZ2hjZWJ3Z3pxZnNnbXhrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgwNTc2MTcsImV4cCI6MjA4MzYzMzYxN30.JFbrt84nne3v2gPfhWiixsd5fWbkVkEWFGC2P2KoqUc';

try {
  const decoded = jwt.decode(sampleToken, { complete: true });
  console.log('JWT Header:', decoded.header);
  console.log('JWT Payload:', decoded.payload);
  console.log('Algorithm:', decoded.header.alg);
  
  // Check if it's ES256 (asymmetric) or HS256 (symmetric)
  if (decoded.header.alg === 'ES256') {
    console.log('✅ This is an ES256 token (asymmetric) - needs JWKS verification');
  } else if (decoded.header.alg === 'HS256') {
    console.log('❌ This is an HS256 token (symmetric) - needs secret verification');
  }
} catch (error) {
  console.error('Error decoding JWT:', error);
}
