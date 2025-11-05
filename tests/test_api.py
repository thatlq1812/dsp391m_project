"""Test FastAPI endpoints.

Skips automatically when the local API server is not running on localhost:8080.
Run the server with:
  python -m uvicorn traffic_api.main:app --port 8080
"""

import json
from datetime import datetime

import pytest
import requests

BASE_URL = "http://localhost:8080"


def _api_available() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/", timeout=0.3)
        return r.status_code in (200, 404)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _api_available(), reason="API server not running on localhost:8080"
)


def test_root():
    """Test root endpoint."""
    print("\n=== Testing / ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_health():
    """Test health check."""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, default=str))


def test_nodes():
    """Test get all nodes."""
    print("\n=== Testing /nodes ===")
    response = requests.get(f"{BASE_URL}/nodes")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total nodes: {len(data)}")
    print(f"First node: {json.dumps(data[0], indent=2, ensure_ascii=False)}")


def test_predict():
    """Test predictions."""
    print("\n=== Testing /predict ===")
    
    payload = {
        "horizons": [1, 3, 6, 12]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Timestamp: {data['timestamp']}")
        print(f"Model version: {data['model_version']}")
        print(f"Inference time: {data['inference_time_ms']:.2f} ms")
        print(f"Total nodes: {len(data['nodes'])}")
        print(f"\nFirst node prediction:")
        print(json.dumps(data['nodes'][0], indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.text}")


def test_specific_node():
    """Test single node prediction."""
    print("\n=== Testing /predict for specific node ===")
    
    payload = {
        "node_ids": ["node-10.737984-106.721606"],
        "horizons": [1, 2, 3]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data['nodes'][0], indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("Testing STMGT Traffic API")
    print("=" * 50)
    
    try:
        test_root()
        test_health()
        test_nodes()
        test_predict()
        test_specific_node()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API!")
        print("Make sure the API is running: python -m uvicorn traffic_api.main:app --port 8080")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
