"""
Quick API starter script for STMGT Traffic Intelligence
Run this to start the API server: python start_api_simple.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Change to project root
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("STMGT Traffic Intelligence API Server")
    print("=" * 60)
    print()
    print("Starting API on http://localhost:8080")
    print("Press Ctrl+C to stop")
    print()
    print("=" * 60)
    
    try:
        # Start uvicorn
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "traffic_api.main:app",
                "--host", "0.0.0.0",
                "--port", "8080",
                "--reload"
            ],
            cwd=project_root,
            check=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down API server...")
    except Exception as e:
        print(f"\nError starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
