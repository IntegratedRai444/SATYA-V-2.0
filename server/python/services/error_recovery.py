"""
Error Recovery Service
Provides enhanced error recovery strategies for the ML backend
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class ErrorRecoveryManager:
    """Manages error recovery strategies for the ML backend"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, Dict[str, Any]] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        
    def register_recovery_handler(self, service_name: str, handler: Callable):
        """Register a custom recovery handler for a service"""
        self.recovery_handlers[service_name] = handler
        logger.info(f"Registered recovery handler for {service_name}")
    
    async def execute_with_recovery(
        self,
        service_name: str,
        operation: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_operation: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Execute an operation with automatic error recovery"""
        
        # Check circuit breaker
        if self._is_circuit_open(service_name):
            logger.warning(f"Circuit breaker open for {service_name}, using fallback")
            if fallback_operation:
                return await fallback_operation(*args, **kwargs)
            raise Exception(f"Service {service_name} is temporarily unavailable")
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success - reset circuit breaker and retry count
                self._reset_circuit_breaker(service_name)
                self.retry_counts[service_name] = 0
                
                return result
                
            except Exception as e:
                last_exception = e
                self._record_error(service_name, e, attempt)
                
                if attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = retry_delay * (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {service_name}, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for {service_name}: {e}")
        
        # All retries failed - try recovery strategies
        recovery_result = await self._attempt_recovery(service_name, last_exception, fallback_operation, *args, **kwargs)
        
        if recovery_result is not None:
            return recovery_result
        
        # If all recovery attempts failed, raise the last exception
        raise last_exception
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            return False
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if datetime.now() > breaker['reset_time']:
                breaker['state'] = 'half_open'
                logger.info(f"Circuit breaker for {service_name} moved to half-open state")
                return False
            return True
        
        return False
    
    def _record_error(self, service_name: str, error: Exception, attempt: int):
        """Record an error for circuit breaker and retry tracking"""
        # Update retry count
        self.retry_counts[service_name] = self.retry_counts.get(service_name, 0) + 1
        
        # Record last error
        self.last_errors[service_name] = {
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now(),
            'attempt': attempt
        }
        
        # Update circuit breaker
        breaker = self.circuit_breakers.get(service_name, {
            'state': 'closed',
            'failure_count': 0,
            'last_failure': None,
            'reset_time': None
        })
        
        breaker['failure_count'] += 1
        breaker['last_failure'] = datetime.now()
        
        # Open circuit if failure threshold exceeded
        if breaker['failure_count'] >= 5:  # 5 failures trigger circuit breaker
            breaker['state'] = 'open'
            breaker['reset_time'] = datetime.now() + timedelta(minutes=5)  # 5 minute timeout
            logger.warning(f"Circuit breaker opened for {service_name} due to repeated failures")
        
        self.circuit_breakers[service_name] = breaker
    
    def _reset_circuit_breaker(self, service_name: str):
        """Reset circuit breaker for a service after successful operation"""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name]['state'] = 'closed'
            self.circuit_breakers[service_name]['failure_count'] = 0
            logger.debug(f"Circuit breaker reset for {service_name}")
    
    async def _attempt_recovery(
        self,
        service_name: str,
        error: Exception,
        fallback_operation: Optional[Callable],
        *args,
        **kwargs
    ) -> Any:
        """Attempt various recovery strategies"""
        
        # Try custom recovery handler first
        if service_name in self.recovery_handlers:
            try:
                logger.info(f"Attempting custom recovery for {service_name}")
                if asyncio.iscoroutinefunction(self.recovery_handlers[service_name]):
                    result = await self.recovery_handlers[service_name](error, *args, **kwargs)
                else:
                    result = self.recovery_handlers[service_name](error, *args, **kwargs)
                
                if result is not None:
                    logger.info(f"Custom recovery successful for {service_name}")
                    return result
            except Exception as recovery_error:
                logger.error(f"Custom recovery failed for {service_name}: {recovery_error}")
        
        # Try fallback operation
        if fallback_operation:
            try:
                logger.info(f"Attempting fallback operation for {service_name}")
                if asyncio.iscoroutinefunction(fallback_operation):
                    result = await fallback_operation(*args, **kwargs)
                else:
                    result = fallback_operation(*args, **kwargs)
                
                logger.info(f"Fallback operation successful for {service_name}")
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback operation failed for {service_name}: {fallback_error}")
        
        # Try graceful degradation
        try:
            result = await self._graceful_degradation(service_name, error, *args, **kwargs)
            if result is not None:
                logger.info(f"Graceful degradation successful for {service_name}")
                return result
        except Exception as degradation_error:
            logger.error(f"Graceful degradation failed for {service_name}: {degradation_error}")
        
        return None
    
    async def _graceful_degradation(
        self,
        service_name: str,
        error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """Attempt graceful degradation for the failed service"""
        
        # Service-specific degradation strategies
        if service_name == 'image_detector':
            # Return a basic analysis result
            return {
                'success': True,
                'authenticity': 'UNCERTAIN',
                'confidence': 0.5,
                'error': f'ML model unavailable: {str(error)}',
                'fallback_used': True,
                'analysis_type': 'graceful_degradation'
            }
        
        elif service_name == 'database':
            # Return cached data or default values
            return {
                'success': True,
                'data': [],
                'cached': True,
                'error': f'Database unavailable: {str(error)}',
                'fallback_used': True
            }
        
        elif service_name == 'cache':
            # Continue without cache
            logger.warning(f"Cache unavailable for {service_name}, continuing without cache")
            return None
        
        # Default graceful degradation
        return {
            'success': False,
            'error': f'Service {service_name} unavailable: {str(error)}',
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get the health status of a service"""
        breaker = self.circuit_breakers.get(service_name, {'state': 'closed', 'failure_count': 0})
        last_error = self.last_errors.get(service_name)
        retry_count = self.retry_counts.get(service_name, 0)
        
        return {
            'service': service_name,
            'circuit_breaker_state': breaker['state'],
            'failure_count': breaker['failure_count'],
            'retry_count': retry_count,
            'last_error': last_error,
            'healthy': breaker['state'] == 'closed' and retry_count < 3
        }
    
    def get_all_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all services"""
        services = set()
        services.update(self.circuit_breakers.keys())
        services.update(self.last_errors.keys())
        services.update(self.retry_counts.keys())
        
        return {service: self.get_service_health(service) for service in services}

# Global error recovery manager
error_recovery = ErrorRecoveryManager()
