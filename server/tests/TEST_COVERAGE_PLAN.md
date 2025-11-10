# Test Coverage Expansion Plan

## Current Status
- ✅ E2E test framework exists
- ✅ Production readiness checks exist
- ⚠️ Limited unit test coverage
- ⚠️ Need more integration tests

## Priority Tests to Add

### High Priority (Core Functionality)

#### 1. API Route Tests
- [ ] `tests/unit/routes/auth.test.ts` - Authentication endpoints
- [ ] `tests/unit/routes/analysis.test.ts` - Analysis endpoints
- [ ] `tests/unit/routes/upload.test.ts` - Upload endpoints
- [ ] `tests/unit/routes/dashboard.test.ts` - Dashboard endpoints

#### 2. Service Tests
- [ ] `tests/unit/services/python-bridge.test.ts` - Python communication
- [ ] `tests/unit/services/websocket-manager.test.ts` - WebSocket management
- [ ] `tests/unit/services/health-monitor.test.ts` - Health monitoring
- [ ] `tests/unit/services/session-manager.test.ts` - Session management

#### 3. Middleware Tests
- [ ] `tests/unit/middleware/auth-middleware.test.ts` - Authentication
- [ ] `tests/unit/middleware/error-handler.test.ts` - Error handling
- [ ] `tests/unit/middleware/input-validation.test.ts` - Input validation

### Medium Priority (Integration)

#### 4. Integration Tests
- [ ] `tests/integration/auth-flow.test.ts` - Complete auth flow
- [ ] `tests/integration/analysis-flow.test.ts` - Analysis workflow
- [ ] `tests/integration/upload-flow.test.ts` - Upload workflow
- [ ] `tests/integration/websocket-flow.test.ts` - WebSocket communication

### Low Priority (Edge Cases)

#### 5. Edge Case Tests
- [ ] Rate limiting behavior
- [ ] Error scenarios
- [ ] Concurrent requests
- [ ] Large file uploads

## Test Template

```typescript
// Example: tests/unit/services/example.test.ts
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { exampleService } from '../../../services/example-service';

describe('ExampleService', () => {
  beforeEach(() => {
    // Setup
  });

  afterEach(() => {
    // Cleanup
  });

  describe('methodName', () => {
    it('should handle success case', async () => {
      // Arrange
      const input = { test: 'data' };
      
      // Act
      const result = await exampleService.methodName(input);
      
      // Assert
      expect(result).toBeDefined();
      expect(result.success).toBe(true);
    });

    it('should handle error case', async () => {
      // Arrange
      const invalidInput = null;
      
      // Act & Assert
      await expect(
        exampleService.methodName(invalidInput)
      ).rejects.toThrow();
    });
  });
});
```

## Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- auth.test.ts

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

## Coverage Goals

- **Target:** 80% code coverage
- **Critical paths:** 100% coverage
- **Services:** 90% coverage
- **Routes:** 85% coverage
- **Middleware:** 90% coverage

## Next Steps

1. Create test files from template
2. Write tests for critical paths first
3. Add integration tests
4. Run coverage reports
5. Iterate until 80% coverage achieved
