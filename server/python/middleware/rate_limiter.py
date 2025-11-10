from flask import request, jsonify
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List
import time

class RateLimiter:
    def __init__(self, requests: int = 10, window: int = 60):
        self.requests = requests  # Number of requests
        self.window = window      # Time window in seconds
        self.access_records: Dict[str, List[float]] = {}

    def limit(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client IP
            client_ip = request.remote_addr or '127.0.0.1'
            
            # Get current timestamp
            current_time = time.time()
            
            # Initialize client's access records if not exists
            if client_ip not in self.access_records:
                self.access_records[client_ip] = []
            
            # Remove timestamps older than the time window
            window_start = current_time - self.window
            self.access_records[client_ip] = [t for t in self.access_records[client_ip] if t > window_start]
            
            # Check if rate limit exceeded
            if len(self.access_records[client_ip]) >= self.requests:
                return jsonify({
                    "status": "error",
                    "message": f"Rate limit exceeded. Try again in {self.window} seconds.",
                    "retry_after": self.window
                }), 429
            
            # Add current request timestamp
            self.access_records[client_ip].append(current_time)
            
            # Proceed with the request
            return f(*args, **kwargs)
        return wrapped

# Create a rate limiter instance (10 requests per minute by default)
rate_limiter = RateLimiter(requests=10, window=60)
