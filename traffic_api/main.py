"""FastAPI Application - Traffic Forecasting API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from traffic_api.config import config
from traffic_api.predictor import STMGTPredictor
from traffic_api.schemas import (
    ErrorResponse,
    HealthResponse,
    NodeInfo,
    PredictionRequest,
    PredictionResponse,
    TrafficCurrentResponse,
    EdgeTraffic,
    RouteRequest,
    RoutePlanResponse,
)

# Authentication and rate limiting
from traffic_api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    Token,
    User,
)
from traffic_api.rate_limit import (
    limiter,
    custom_rate_limit_handler,
    RateLimitMiddleware,
)
from slowapi.errors import RateLimitExceeded

# Create FastAPI app
app = FastAPI(
    title="STMGT Traffic Forecasting API",
    description="Real-time traffic speed forecasting for Ho Chi Minh City with JWT authentication",
    version="1.0.0",
    docs_url="/api/docs",  # Swagger UI
    redoc_url="/api/redoc",  # ReDoc
)

# Add rate limiter state
app.state.limiter = limiter

# Add rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

# Add rate limit middleware
app.add_middleware(RateLimitMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print(f"Static files mounted from {static_dir}")
else:
    print(f"Static directory not found: {static_dir}")

# Global predictor instance
predictor: Optional[STMGTPredictor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    
    if config.model_checkpoint is None:
        print("WARNING: No model checkpoint found!")
        return
    
    if config.data_path is None or not config.data_path.exists():
        print(f"WARNING: Data file not found: {config.data_path}")
        return
    
    try:
        predictor = STMGTPredictor(
            checkpoint_path=config.model_checkpoint,
            data_path=config.data_path,
            device=config.device,
        )
        print("Predictor initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize predictor: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/auth/login", response_model=Token, tags=["Authentication"])
@limiter.limit("10/minute")  # Strict rate limit for login
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """
    Login with username and password to get JWT token.
    
    **Rate limit:** 10 requests/minute
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/auth/login" \\
         -d "username=demo&password=demo123"
    ```
    """
    user = authenticate_user(username, password)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=User, tags=["Authentication"])
@limiter.limit("60/minute")
async def get_me(request: Request, current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    **Requires:** JWT token in Authorization header
    
    **Example:**
    ```bash
    curl -H "Authorization: Bearer YOUR_TOKEN" \\
         http://localhost:8000/api/auth/me
    ```
    """
    return current_user


# ============================================================================
# Public Endpoints (No Authentication Required)
# ============================================================================

@app.get("/", response_class=FileResponse)
@limiter.limit("200/minute")
async def root(request: Request):
    """Serve web interface."""
    static_dir = Path(__file__).parent / "static"
    traffic_intelligence_path = static_dir / "traffic_intelligence.html"
    
    if traffic_intelligence_path.exists():
        return FileResponse(traffic_intelligence_path)
    else:
        # Fallback to JSON response if static files not available
        return {
            "service": "STMGT Traffic Forecasting API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "nodes": "/nodes",
                "predict": "/predict",
            },
        }


@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.limit("200/minute")
async def health_check(request: Request):
    """
    Health check endpoint (public, no authentication required).
    
    **Rate limit:** 200 requests/minute
    """
    return HealthResponse(
        status="healthy" if predictor else "no_model",
        model_loaded=predictor is not None,
        model_checkpoint=str(config.model_checkpoint) if config.model_checkpoint else None,
        device=config.device,
        timestamp=datetime.now(),
    )


@app.get("/api/route-geometries")
async def get_route_geometries():
    """Get pre-computed route geometries from Google Maps (optional)."""
    geometries_file = Path(__file__).parent.parent / "cache" / "route_geometries.json"
    
    if not geometries_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Route geometries not found. Run scripts/data/fetch_route_geometries.py first"
        )
    
    return FileResponse(geometries_file, media_type="application/json")


@app.get("/api/edge-geometries")
async def get_edge_geometries():
    """Get edge coordinates extracted from training data (straight lines)."""
    geometries_file = Path(__file__).parent.parent / "cache" / "edge_coordinates.json"
    
    if not geometries_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Edge coordinates not found. Run scripts/data/extract_edge_coordinates.py first"
        )
    
    return FileResponse(geometries_file, media_type="application/json")


@app.get("/nodes", response_model=list[NodeInfo])
async def get_nodes():
    """Get all traffic nodes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        nodes = predictor.get_nodes()
        return [NodeInfo(**node) for node in nodes]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nodes: {str(e)}")


@app.get("/nodes/{node_id}", response_model=NodeInfo)
async def get_node(node_id: str):
    """Get specific node information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    node = predictor.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return NodeInfo(**node)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
# @limiter.limit("60/minute")  # Temporarily disabled due to slowapi parameter issues
async def predict(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)  # Require authentication
):
    """
    Generate traffic predictions (requires authentication).
    
    **Requires:** JWT token in Authorization header  
    **Rate limit:** 60 requests/minute
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/predict" \\
         -H "Authorization: Bearer YOUR_TOKEN" \\
         -H "Content-Type: application/json" \\
         -d '{"node_ids": ["node_1"], "horizons": [1, 3, 6]}'
    ```
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        result = predictor.predict(
            timestamp=request.timestamp,
            node_ids=request.node_ids,
            horizons=request.horizons,
        )
        return PredictionResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/traffic/current", response_model=TrafficCurrentResponse)
async def get_current_traffic():
    """Get current traffic status for all edges with color coding."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Get latest data
        current_data = predictor.get_current_traffic()
        
        # Convert speeds to colors
        edges = []
        for edge in current_data:
            speed = edge['speed_kmh']
            color, category = _speed_to_color(speed)
            
            edges.append(EdgeTraffic(
                edge_id=edge['edge_id'],
                node_a_id=edge['node_a_id'],
                node_b_id=edge['node_b_id'],
                speed_kmh=speed,
                color=color,
                color_category=category,
                timestamp=edge['timestamp'],
                lat_a=edge['lat_a'],
                lon_a=edge['lon_a'],
                lat_b=edge['lat_b'],
                lon_b=edge['lon_b']
            ))
        
        return TrafficCurrentResponse(
            edges=edges,
            timestamp=datetime.now(),
            total_edges=len(edges)
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get traffic: {str(e)}")


@app.post("/api/route/plan", response_model=RoutePlanResponse)
async def plan_route(request: RouteRequest):
    """Plan optimal route from start to end node."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        departure_time = request.departure_time or datetime.now()
        
        # Get 3 route options
        routes = predictor.plan_routes(
            start_node_id=request.start_node_id,
            end_node_id=request.end_node_id,
            departure_time=departure_time
        )
        
        return RoutePlanResponse(
            start_node_id=request.start_node_id,
            end_node_id=request.end_node_id,
            departure_time=departure_time,
            routes=routes,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Route planning failed: {str(e)}")


@app.get("/api/predict/{edge_id}")
async def predict_edge(edge_id: str, horizon: int = 12):
    """Get prediction for specific edge."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        prediction = predictor.predict_edge(edge_id, horizon)
        return prediction
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _speed_to_color(speed_kmh: float) -> tuple[str, str]:
    """Convert speed to gradient color."""
    if speed_kmh >= 50:
        return "#0066FF", "blue"  # Very smooth
    elif speed_kmh >= 40:
        return "#00CC00", "green"  # Smooth
    elif speed_kmh >= 30:
        return "#90EE90", "light_green"  # Normal
    elif speed_kmh >= 20:
        return "#FFD700", "yellow"  # Slow
    elif speed_kmh >= 10:
        return "#FF8800", "orange"  # Congested
    else:
        return "#FF0000", "red"  # Heavy traffic


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now(),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "traffic_api.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
