"""
Quick API Starter for Windows
Run this directly with Python to start the API server
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from traffic_api.main import app

if __name__ == "__main__":
    print("=" * 60)
    print("STMGT V3 API Server")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"API Endpoint: http://localhost:8080")
    print(f"API Docs: http://localhost:8080/docs")
    print("=" * 60)
    print("\nStarting server... (Press Ctrl+C to stop)\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
