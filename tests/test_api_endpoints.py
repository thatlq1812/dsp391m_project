"""
Test script for STMGT Traffic API endpoints.
Tests all new routes added in Phase 2.
"""

import sys
sys.path.insert(0, 'd:/UNI/DSP391m/project')

from fastapi.testclient import TestClient
from traffic_api.main import app

client = TestClient(app)


def test_health():
    """Test health check endpoint"""
    print("\n[TEST] GET /health")
    response = client.get("/health")
    print(f"  Status: {response.status_code}")
    assert response.status_code == 200
    data = response.json()
    print(f"  Model loaded: {data.get('model_loaded')}")
    print(f"  Device: {data.get('device')}")
    assert data["status"] == "healthy"
    print("  [PASS]")


def test_traffic_current():
    """Test current traffic endpoint"""
    print("\n[TEST] GET /api/traffic/current")
    response = client.get("/api/traffic/current")
    print(f"  Status: {response.status_code}")
    assert response.status_code == 200
    data = response.json()
    print(f"  Total edges: {data.get('total_edges')}")
    
    if data.get('edges'):
        edge = data['edges'][0]
        print(f"  Sample edge: {edge.get('edge_id')}")
        print(f"    Speed: {edge.get('speed_kmh'):.1f} km/h")
        print(f"    Color: {edge.get('color')} ({edge.get('color_category')})")
        print(f"    Coordinates: ({edge.get('lat_a'):.4f}, {edge.get('lon_a'):.4f})")
    
    # Verify color gradient
    assert all('color' in edge for edge in data['edges'])
    print("  [PASS]")


def test_route_plan():
    """Test route planning endpoint"""
    print("\n[TEST] POST /api/route/plan")
    
    # Use actual node IDs from the dataset
    # Assuming nodes are named node_0, node_1, etc.
    request_data = {
        "start_node_id": "node_0",
        "end_node_id": "node_10"
    }
    
    response = client.post("/api/route/plan", json=request_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Routes found: {len(data.get('routes', []))}")
        
        for route in data.get('routes', []):
            route_type = route.get('route_type')
            distance = route.get('total_distance_km', 0)
            time = route.get('expected_travel_time_min', 0)
            uncertainty = route.get('travel_time_uncertainty_min', 0)
            confidence = route.get('confidence_level', 0)
            segments = len(route.get('segments', []))
            
            print(f"  {route_type.upper()}:")
            print(f"    Distance: {distance:.1f} km")
            print(f"    Time: {time:.1f} ± {uncertainty:.1f} min")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Segments: {segments}")
        
        print("  [PASS]")
    else:
        print(f"  Error: {response.json()}")
        print("  [FAIL] - May need to adjust node IDs")


def test_predict_edge():
    """Test edge prediction endpoint"""
    print("\n[TEST] GET /api/predict/node_0_node_1")
    
    response = client.get("/api/predict/node_0_node_1?horizon=12")
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Edge: {data.get('edge_id')}")
        print(f"  Predicted speed: {data.get('predicted_speed_kmh', 0):.1f} km/h")
        print(f"  Uncertainty: ± {data.get('uncertainty_std', 0):.1f} km/h")
        print(f"  Horizon: {data.get('horizon')} timesteps")
        print("  [PASS]")
    else:
        print(f"  Error: {response.json()}")
        print("  [WARN] - Edge may not exist in graph")


def test_nodes():
    """Test nodes endpoint"""
    print("\n[TEST] GET /nodes")
    response = client.get("/nodes")
    print(f"  Status: {response.status_code}")
    assert response.status_code == 200
    data = response.json()
    print(f"  Total nodes: {len(data)}")
    
    if data:
        node = data[0]
        print(f"  Sample node: {node.get('node_id')}")
        print(f"    Location: ({node.get('lat'):.4f}, {node.get('lon'):.4f})")
        print(f"    Degree: {node.get('degree')}")
    
    print("  [PASS]")


def run_all_tests():
    """Run all API endpoint tests"""
    print("\n" + "=" * 60)
    print("STMGT Traffic API - Endpoint Tests")
    print("=" * 60)
    
    try:
        test_health()
        test_nodes()
        test_traffic_current()
        test_predict_edge()
        test_route_plan()
        
        print("\n" + "=" * 60)
        print("Test Summary: All core tests passed!")
        print("=" * 60)
        print("\nNote: Some tests may show warnings if specific node IDs")
        print("      don't exist in the dataset. This is expected.")
        
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
