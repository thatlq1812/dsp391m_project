"""Quick API test without running server (direct Python test)."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traffic_api.predictor import STMGTPredictor
from traffic_api.config import config
from datetime import datetime
import json

def test_predictor():
    """Test STMGT predictor directly."""
    print("=" * 70)
    print("TESTING STMGT PREDICTOR (Direct Python Test)")
    print("=" * 70)
    
    # Check paths
    print("\n[1] Checking paths...")
    print(f"Model checkpoint: {config.model_checkpoint}")
    print(f"Data path: {config.data_path}")
    print(f"Device: {config.device}")
    
    if config.model_checkpoint is None or not config.model_checkpoint.exists():
        print("ERROR: Model checkpoint not found!")
        return
    
    if config.data_path is None or not config.data_path.exists():
        print("ERROR: Data file not found!")
        return
    
    # Initialize predictor
    print("\n[2] Initializing predictor...")
    try:
        predictor = STMGTPredictor(
            checkpoint_path=config.model_checkpoint,
            data_path=config.data_path,
            device=config.device,
        )
        print("✓ Predictor initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize predictor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test get_nodes
    print("\n[3] Testing get_nodes()...")
    nodes = predictor.get_nodes()
    print(f"✓ Total nodes: {len(nodes)}")
    print(f"✓ First node: {nodes[0]['node_id']}")
    print(f"   - Location: ({nodes[0]['lat']}, {nodes[0]['lon']})")
    print(f"   - Intersection: {nodes[0]['intersection_name']}")
    
    # Test predict
    print("\n[4] Testing predict()...")
    try:
        result = predictor.predict(
            timestamp=datetime.now(),
            node_ids=[nodes[0]['node_id'], nodes[1]['node_id']],  # Test with 2 nodes
            horizons=[1, 3, 6, 12],
        )
        print(f"✓ Prediction successful!")
        print(f"   - Timestamp: {result['timestamp']}")
        print(f"   - Forecast time: {result['forecast_time']}")
        print(f"   - Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"   - Nodes predicted: {len(result['nodes'])}")
        
        # Show first node prediction
        print(f"\n[5] Sample prediction (first node):")
        node_pred = result['nodes'][0]
        print(f"   Node: {node_pred['node_id']}")
        print(f"   Current speed: {node_pred['current_speed']:.2f} km/h")
        print(f"   Forecasts:")
        for fc in node_pred['forecasts']:
            print(f"     - {fc['horizon_minutes']}min: {fc['mean']:.2f} ± {fc['std']:.2f} km/h "
                  f"({fc['lower_80']:.2f} - {fc['upper_80']:.2f})")
        
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test predict all nodes
    print("\n[6] Testing predict() for all nodes...")
    try:
        result_all = predictor.predict(horizons=[1, 6, 12])
        print(f"✓ Predicted for {len(result_all['nodes'])} nodes")
        print(f"   - Inference time: {result_all['inference_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"ERROR: Full prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run API server: conda run -n dsp python -m uvicorn traffic_api.main:app --port 8080")
    print("2. Test API endpoints: python test_api.py")
    print("3. View Swagger docs: http://localhost:8080/docs")


if __name__ == "__main__":
    test_predictor()
