"""
Custom exceptions and error handling utilities
"""
from typing import Optional
from datetime import datetime, timedelta

class MistralAPIError(Exception):
    """Base exception for Mistral API errors"""
    pass

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass

class ToolExecutionError(Exception):
    """Raised when a tool execution fails"""
    pass

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API calls
    """
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Record a successful call"""
        if self.state == "half_open":
            self.state = "closed"
        self.failures = 0
    
    def record_failure(self):
        """Record a failed call"""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.failures >= self.threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self.state = "half_open"
                    return True
            return False
        
        # half_open state
        return True
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state
