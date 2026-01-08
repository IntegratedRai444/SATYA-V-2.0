# API Client Migration Guide

This guide helps you migrate from the old API client to the new enhanced API client.

## Changes Overview

1. **New Import Path**
   - Old: `import apiClient from '@/lib/api'`
   - New: `import { apiClient } from '@/lib/api'`

2. **Service-Based Architecture**
   - Instead of using `apiClient` directly, use the appropriate service:
     ```typescript
     // Old
     import apiClient from '@/lib/api';
     const response = await apiClient.post('/auth/login', { email, password });
     
     // New
     import { authService } from '@/lib/api';
     const response = await authService.login({ email, password });
     ```

3. **Available Services**
   - `authService`: Authentication related operations
   - `analysisService`: Media analysis operations
   - `userService`: User profile and settings

4. **Error Handling**
   - The new client provides consistent error handling
   - All errors include a `message` and `code`
   - Network errors include `isNetworkError: true`

## Migration Steps

1. Update imports to use the new service-based approach
2. Replace direct API calls with service method calls
3. Update error handling to work with the new error format
4. Remove any manual token handling (now handled automatically)

## Example Migrations

### Authentication

```typescript
// Old
const login = async (email: string, password: string) => {
  try {
    const response = await apiClient.post('/auth/login', { email, password });
    localStorage.setItem('token', response.data.token);
    return response.data.user;
  } catch (error) {
    console.error('Login failed', error);
    throw error;
  }
};

// New
const login = async (email: string, password: string) => {
  try {
    const response = await authService.login({ email, password });
    return response.user;
  } catch (error) {
    console.error('Login failed', error);
    throw error;
  }
};
```

### File Upload

```typescript
// Old
const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post('/analysis/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// New
const uploadFile = async (file: File) => {
  return await analysisService.analyzeImage(file, {
    onUploadProgress: (progress) => {
      console.log(`Upload progress: ${progress}%`);
    }
  });
};
```

## Deprecation Notice

The old API client has been moved to `src/lib/api.old.ts` and will be removed in a future version. Please update your code to use the new service-based approach.
