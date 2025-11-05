"""FastAPI Application - Traffic Forecasting API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from traffic_api.config import config
from traffic_api.predictor import STMGTPredictor
from traffic_api.schemas import (
    ErrorResponse,
    HealthResponse,
    NodeInfo,
    PredictionRequest,
    PredictionResponse,
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


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
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
