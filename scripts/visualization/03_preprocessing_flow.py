"""
Generate Figure 3: Data Preprocessing Pipeline Flow Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import save_figure, FIGURE_DIR

def generate_fig3_preprocessing_flow():
    """Figure 3: Data Preprocessing Flow Diagram"""
    print("Generating Figure 3: Preprocessing Flow...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define box styles
    box_style = "round,pad=0.1"
    
    # Colors
    input_color = '#E8F4F8'  # Light blue
    process_color = '#FFE5B4'  # Peach
    output_color = '#D4EDDA'  # Light green
    
    # Helper function to add a box
    def add_box(x, y, width, height, text, color):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle=box_style, edgecolor='black', 
                            facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, 
               weight='bold', wrap=True)
    
    # Helper function to add arrow
    def add_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 11.5, 'Data Preprocessing Pipeline', 
           ha='center', fontsize=16, weight='bold')
    
    # Layer 1: Input Data
    add_box(5, 10.5, 3, 0.6, 'Raw GPS Trajectory Data\n(Google Maps API)', input_color)
    add_arrow(5, 10.2, 5, 9.8)
    
    # Layer 2: Route Processing
    add_box(2.5, 9.3, 2.5, 0.6, 'Extract Routes\n(Polyline Decode)', process_color)
    add_box(7.5, 9.3, 2.5, 0.6, 'Weather Data\n(OpenWeather API)', input_color)
    add_arrow(2.5, 9.0, 2.5, 8.3)
    add_arrow(7.5, 9.0, 7.5, 8.3)
    
    # Layer 3: Graph Construction
    add_box(2.5, 7.8, 2.5, 0.6, 'Node Detection\n(OSM Intersections)', process_color)
    add_box(7.5, 7.8, 2.5, 0.6, 'Weather Matching\n(Timestamp Align)', process_color)
    add_arrow(2.5, 7.5, 2.5, 6.8)
    add_arrow(7.5, 7.5, 7.5, 6.8)
    
    # Layer 4: Edge Creation
    add_box(2.5, 6.3, 2.5, 0.6, 'Edge Creation\n(Node Pairs)', process_color)
    add_box(7.5, 6.3, 2.5, 0.6, 'Feature Extraction\n(Temp, Wind, Precip)', process_color)
    add_arrow(2.5, 6.0, 5, 5.3)
    add_arrow(7.5, 6.0, 5, 5.3)
    
    # Layer 5: Merge
    add_box(5, 4.8, 3, 0.6, 'Merge Spatial + Weather Features', process_color)
    add_arrow(5, 4.5, 5, 4.0)
    
    # Layer 6: Feature Engineering
    add_box(5, 3.5, 3, 0.6, 'Feature Engineering\n(Hour, DoW, Cyclical)', process_color)
    add_arrow(5, 3.2, 5, 2.7)
    
    # Layer 7: Data Augmentation
    add_box(5, 2.2, 3, 0.6, 'Data Augmentation\n(Extreme Weather)', process_color)
    add_arrow(5, 1.9, 5, 1.4)
    
    # Layer 8: Normalization
    add_box(5, 0.9, 3, 0.6, 'Z-score Normalization', process_color)
    add_arrow(5, 0.6, 5, 0.1)
    
    # Layer 9: Output
    add_box(5, -0.4, 3, 0.6, 'Final Dataset\n(205,920 samples)', output_color)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input Data'),
        mpatches.Patch(facecolor=process_color, edgecolor='black', label='Processing Step'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'fig03_preprocessing_flow')

def main():
    """Generate preprocessing flow figure"""
    print(f"Output directory: {FIGURE_DIR}\n")
    generate_fig3_preprocessing_flow()
    print(f"\nPreprocessing flow figure generated in: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
