"""
Master script to generate all figures for final report

Generates all 17 figures required for the DS Capstone report.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils import FIGURE_DIR

def main():
    print("="*60)
    print("GENERATING ALL FIGURES FOR FINAL REPORT")
    print("="*60)
    print(f"Output directory: {FIGURE_DIR}\n")
    
    # Import all figure generation modules (direct import)
    print("\n[1/4] Generating Data Figures (1-2, 4)...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_figs", Path(__file__).parent / "01_data_figures.py")
        data_figs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_figs)
        data_figs.main()
    except Exception as e:
        print(f"  ⚠️  Error in data figures: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[1.5/4] Generating Preprocessing Flow (3)...")
    try:
        spec = importlib.util.spec_from_file_location("prep_flow", Path(__file__).parent / "03_preprocessing_flow.py")
        prep_flow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep_flow)
        prep_flow.main()
    except Exception as e:
        print(f"  ⚠️  Error in preprocessing flow: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[2/4] Generating EDA Figures (5-10)...")
    try:
        spec = importlib.util.spec_from_file_location("eda_figs", Path(__file__).parent / "02_eda_figures.py")
        eda_figs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eda_figs)
        eda_figs.main()
    except Exception as e:
        print(f"  ⚠️  Error in EDA figures: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[3/4] Generating Results Figures (13-17)...")
    try:
        spec = importlib.util.spec_from_file_location("results_figs", Path(__file__).parent / "04_results_figures.py")
        results_figs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(results_figs)
        results_figs.main()
    except Exception as e:
        print(f"  ⚠️  Error in results figures: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\n✓ All figures saved to: {FIGURE_DIR}")
    print("\nFigures generated:")
    print("  - Fig 1-4: Data & Preprocessing")
    print("  - Fig 5-10: EDA")
    print("  - Fig 13-17: Results")
    print("\nNote: Fig 11-12 (architecture diagrams) need manual creation")

if __name__ == "__main__":
    main()
