"""FastAPI Application - Traffic Forecasting API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
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

# Create FastAPI app
app = FastAPI(
    title="STMGT Traffic Forecasting API",
    description="Real-time traffic speed forecasting for Ho Chi Minh City",
    version="0.1.0",
)

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


@app.get("/", response_class=FileResponse)
async def root():
    """Serve web interface."""
    static_dir = Path(__file__).parent / "static"
    traffic_intelligence_path = static_dir / "traffic_intelligence.html"
    
    if traffic_intelligence_path.exists():
        return FileResponse(traffic_intelligence_path)
    else:
        # Fallback to JSON response if static files not available
        return {
            "service": "STMGT Traffic Forecasting API",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "nodes": "/nodes",
                "predict": "/predict",
            },
        }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor else "no_model",
        model_loaded=predictor is not None,
        model_checkpoint=str(config.model_checkpoint) if config.model_checkpoint else None,
        device=config.device,
        timestamp=datetime.now(),
    )


@app.get("/api/route-geometries")
async def get_route_geometries():
    """Get pre-computed route geometries."""
    geometries_file = Path(__file__).parent.parent / "cache" / "route_geometries.json"
    
    if not geometries_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Route geometries not found. Run scripts/data/fetch_route_geometries.py first"
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest = PredictionRequest()):
    """Generate traffic predictions."""
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
