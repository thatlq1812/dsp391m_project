"""
Simple test for visualization scripts

Run this to verify figure generation works before running full pipeline.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from scripts.visualization import utils
        print("  ✓ utils")
    except Exception as e:
        print(f"  ✗ utils: {e}")
        return False
    
    try:
        import scripts.visualization.utils as viz_utils
        print(f"  ✓ Figure directory: {viz_utils.FIGURE_DIR}")
    except Exception as e:
        print(f"  ✗ Figure directory: {e}")
    
    return True

def test_data_loading():
    """Test that data can be loaded"""
    print("\nTesting data loading...")
    
    try:
        from scripts.visualization.utils import load_parquet_data
        df = load_parquet_data()
        print(f"  ✓ Loaded data: {len(df)} samples")
        print(f"  ✓ Columns: {df.columns.tolist()[:5]}...")
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False

def test_simple_figure():
    """Test generating one simple figure"""
    print("\nTesting simple figure generation...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scripts.visualization.utils import save_figure
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Figure")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        save_figure(fig, "test_figure")
        print("  ✓ Test figure generated successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Figure generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("VISUALIZATION SCRIPTS TEST")
    print("="*60)
    print()
    
    results = {
        'imports': test_imports(),
        'data_loading': test_data_loading(),
        'figure_generation': test_simple_figure()
    }
    
    print()
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("✓ All tests passed! Ready to generate figures.")
        print("\nRun: python scripts/visualization/generate_all_figures.py")
    else:
        print("✗ Some tests failed. Fix issues before generating figures.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
