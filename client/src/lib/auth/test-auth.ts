import { getAccessToken } from './getAccessToken';
import { supabase } from '../supabaseSingleton';

export const testAuthPipeline = async () => {
  console.log('🧪 Testing Authentication Pipeline...');
  
  try {
    // 1. Check if Supabase session exists
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error) {
      console.error('❌ Supabase session error:', error);
      return false;
    }
    
    if (!session) {
      console.error('❌ No Supabase session found');
      return false;
    }
    
    console.log('✅ Supabase session exists:', {
      userId: session.user.id,
      email: session.user.email,
      hasAccessToken: !!session.access_token,
      tokenLength: session.access_token?.length
    });
    
    // 2. Test getAccessToken function
    const token = await getAccessToken();
    if (!token) {
      console.error('❌ getAccessToken returned null');
      return false;
    }
    
    console.log('✅ getAccessToken works:', {
      hasToken: !!token,
      tokenLength: token.length,
      tokenPreview: token.substring(0, 20) + '...'
    });
    
    // 3. Verify token format (JWT should have 3 parts)
    const parts = token.split('.');
    if (parts.length !== 3) {
      console.error('❌ Invalid JWT format - expected 3 parts, got', parts.length);
      return false;
    }
    
    console.log('✅ JWT format valid (3 parts)');
    
    // 4. Decode JWT payload (without verification)
    const payload = JSON.parse(atob(parts[1]));
    console.log('✅ JWT payload decoded:', {
      userId: payload.sub,
      exp: new Date(payload.exp * 1000).toISOString(),
      iat: new Date(payload.iat * 1000).toISOString(),
      role: payload.user_metadata?.role
    });
    
    // Check if token is expired
    if (Date.now() / 1000 > payload.exp) {
      console.error('❌ Token is expired');
      return false;
    }
    
    console.log('✅ Token is valid and not expired');
    
    return true;
    
  } catch (error) {
    console.error('❌ Auth pipeline test failed:', error);
    return false;
  }
};

// Auto-run test in development
if (import.meta.env.DEV) {
  testAuthPipeline().then(success => {
    console.log(`🎯 Auth pipeline test ${success ? 'PASSED' : 'FAILED'}`);
  });
}
