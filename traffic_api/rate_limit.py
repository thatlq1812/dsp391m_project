"""
Rate Limiting Middleware for STMGT Traffic API
Protects against abuse and ensures fair resource allocation.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from typing import Callable
import time


# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "1000/hour"],
    headers_enabled=True,  # Add rate limit info to response headers
)


def get_limiter():
    """Get the limiter instance"""
    return limiter


# Custom rate limit handler
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded.
    Returns JSON response with helpful information.
    """
    from fastapi.responses import JSONResponse
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {exc.detail}",
            "retry_after": getattr(exc, "retry_after", 60),
            "hint": "Consider using API key for higher limits"
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60))
        }
    )


# Rate limit decorators for common use cases

def rate_limit_basic(func: Callable) -> Callable:
    """
    Basic rate limit: 100 requests/minute
    
    Usage:
        @app.get("/api/data")
        @rate_limit_basic
        async def get_data():
            return {"data": "..."}
    """
    return limiter.limit("100/minute")(func)


def rate_limit_strict(func: Callable) -> Callable:
    """
    Strict rate limit: 30 requests/minute for expensive operations
    
    Usage:
        @app.post("/api/predict")
        @rate_limit_strict
        async def predict():
            # Expensive prediction operation
    """
    return limiter.limit("30/minute")(func)


def rate_limit_generous(func: Callable) -> Callable:
    """
    Generous rate limit: 200 requests/minute for lightweight operations
    
    Usage:
        @app.get("/health")
        @rate_limit_generous
        async def health_check():
            return {"status": "ok"}
    """
    return limiter.limit("200/minute")(func)


class RateLimitMiddleware:
    """
    Middleware to track request rates and add headers.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add custom headers
                headers = list(message.get("headers", []))
                
                # Add processing time
                process_time = time.time() - start_time
                headers.append((b"X-Process-Time", f"{process_time:.3f}".encode()))
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# IP whitelist for internal/admin access (bypass rate limits)
WHITELIST_IPS = {
    "127.0.0.1",
    "::1",
    "localhost",
}


def is_whitelisted(request: Request) -> bool:
    """Check if request IP is whitelisted"""
    client_ip = request.client.host if request.client else None
    return client_ip in WHITELIST_IPS


# Custom key function that considers authentication
def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key based on user or IP.
    Authenticated users get separate buckets.
    """
    # Check if user is authenticated
    auth_header = request.headers.get("Authorization")
    
    if auth_header and auth_header.startswith("Bearer "):
        # Use token hash as key (unique per user)
        token = auth_header.split(" ")[1]
        return f"user:{hash(token)}"
    
    # Fall back to IP address
    return get_remote_address(request)


# Advanced limiter with user-aware rate limiting
advanced_limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=["100/minute"],
    headers_enabled=True,
)
