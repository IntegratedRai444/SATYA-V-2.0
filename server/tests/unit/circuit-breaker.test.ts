import { CircuitBreaker } from '../../utils/circuitBreaker';

describe('CircuitBreaker', () => {
  let circuitBreaker: CircuitBreaker;
  const successFn = () => Promise.resolve('success');
  const failureFn = () => Promise.reject(new Error('Failed'));

  beforeEach(() => {
    jest.useFakeTimers();
    circuitBreaker = new CircuitBreaker({
      failureThreshold: 3,
      successThreshold: 2,
      timeout: 5000 // 5 seconds
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should be in CLOSED state initially', () => {
    expect(circuitBreaker.getState()).toBe('CLOSED');
  });

  it('should execute successful function calls', async () => {
    const result = await circuitBreaker.execute(successFn);
    expect(result).toBe('success');
    expect(circuitBreaker.getState()).toBe('CLOSED');
  });

  it('should trip to OPEN state after threshold failures', async () => {
    // First 2 failures (threshold is 3)
    for (let i = 0; i < 2; i++) {
      await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
      expect(circuitBreaker.getState()).toBe('CLOSED');
    }

    // 3rd failure should trip the circuit
    await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
    expect(circuitBreaker.getState()).toBe('OPEN');

    // Next call should fail fast
    await expect(circuitBreaker.execute(successFn)).rejects.toThrow('Circuit breaker is open');
  });

  it('should transition to HALF_OPEN after timeout', async () => {
    // Trip the circuit
    for (let i = 0; i < 3; i++) {
      await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
    }
    expect(circuitBreaker.getState()).toBe('OPEN');

    // Fast-forward time just before timeout
    jest.advanceTimersByTime(4999);
    expect(circuitBreaker.getState()).toBe('OPEN');

    // After timeout, should be HALF_OPEN
    jest.advanceTimersByTime(2);
    await expect(circuitBreaker.execute(successFn)).resolves.toBe('success');
    expect(circuitBreaker.getState()).toBe('HALF_OPEN');
  });

  it('should reset to CLOSED after success threshold in HALF_OPEN', async () => {
    // Trip the circuit and advance to HALF_OPEN
    for (let i = 0; i < 3; i++) {
      await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
    }
    jest.advanceTimersByTime(5001);

    // First success in HALF_OPEN
    await expect(circuitBreaker.execute(successFn)).resolves.toBe('success');
    expect(circuitBreaker.getState()).toBe('HALF_OPEN');

    // Second success should close the circuit
    await expect(circuitBreaker.execute(successFn)).resolves.toBe('success');
    expect(circuitBreaker.getState()).toBe('CLOSED');
  });

  it('should trip back to OPEN on failure in HALF_OPEN', async () => {
    // Trip the circuit and advance to HALF_OPEN
    for (let i = 0; i < 3; i++) {
      await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
    }
    jest.advanceTimersByTime(5001);

    // First call in HALF_OPEN fails
    await expect(circuitBreaker.execute(failureFn)).rejects.toThrow('Failed');
    expect(circuitBreaker.getState()).toBe('OPEN');
  });
});
