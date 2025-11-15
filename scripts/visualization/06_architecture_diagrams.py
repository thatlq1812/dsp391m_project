"""
Generate architecture diagrams (Figures 11-12)

Fig 11: STMGT Overall Architecture Block Diagram
Fig 12: Attention Mechanism Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import save_figure, FIGURE_DIR

def generate_fig11_stmgt_architecture():
    """Figure 11: STMGT Overall Architecture Block Diagram"""
    print("Generating Figure 11: STMGT Architecture...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    input_color = '#E8F4F8'
    spatial_color = '#FFE5E5'
    temporal_color = '#E5F5E5'
    fusion_color = '#FFF4E5'
    output_color = '#F0E8FF'
    
    # Title
    ax.text(7, 9.5, 'STMGT Architecture Overview', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # ==================== INPUT LAYER ====================
    input_box = FancyBboxPatch((0.5, 8), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.4, 'Input Embedding', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, 8.1, 'Speed + Weather → D=96', ha='center', va='center', fontsize=9)
    
    # ==================== STMGT BLOCKS (3 BLOCKS) ====================
    block_y_start = 5.5
    block_height = 2
    
    for block_idx in range(3):
        block_y = block_y_start - (block_idx * 0.0)  # All at same level for parallel
        
        # Block container
        block_box = mpatches.FancyBboxPatch((0.2, block_y - 0.2), 13.6, block_height + 0.4,
                                            boxstyle="round,pad=0.05",
                                            edgecolor='gray', facecolor='white', 
                                            linewidth=1.5, linestyle='--', alpha=0.3)
        ax.add_patch(block_box)
        ax.text(0.5, block_y + block_height + 0.1, f'Block {block_idx + 1}', 
                fontsize=9, fontstyle='italic', color='gray')
    
    # Spatial Branch (left)
    spatial_x = 1.5
    spatial_y = block_y_start + 0.2
    
    spatial_box = FancyBboxPatch((spatial_x, spatial_y), 3.5, 1.5,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkred', facecolor=spatial_color, linewidth=2)
    ax.add_patch(spatial_box)
    ax.text(spatial_x + 1.75, spatial_y + 1.2, 'Spatial Branch', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(spatial_x + 1.75, spatial_y + 0.85, 'GATv2', ha='center', va='center', fontsize=10)
    ax.text(spatial_x + 1.75, spatial_y + 0.55, '4 attention heads', ha='center', va='center', fontsize=8)
    ax.text(spatial_x + 1.75, spatial_y + 0.25, 'Graph: 62 nodes, 144 edges', 
            ha='center', va='center', fontsize=8)
    
    # Temporal Branch (right)
    temporal_x = 9
    temporal_y = block_y_start + 0.2
    
    temporal_box = FancyBboxPatch((temporal_x, temporal_y), 3.5, 1.5,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='darkgreen', facecolor=temporal_color, linewidth=2)
    ax.add_patch(temporal_box)
    ax.text(temporal_x + 1.75, temporal_y + 1.2, 'Temporal Branch',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(temporal_x + 1.75, temporal_y + 0.85, 'Transformer', ha='center', va='center', fontsize=10)
    ax.text(temporal_x + 1.75, temporal_y + 0.55, '4 attention heads', ha='center', va='center', fontsize=8)
    ax.text(temporal_x + 1.75, temporal_y + 0.25, 'Seq: 12 timesteps', 
            ha='center', va='center', fontsize=8)
    
    # Gated Fusion (center bottom)
    fusion_x = 5.5
    fusion_y = block_y_start - 0.8
    
    fusion_box = FancyBboxPatch((fusion_x, fusion_y), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='darkorange', facecolor=fusion_color, linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(fusion_x + 1.5, fusion_y + 0.55, 'Gated Fusion', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(fusion_x + 1.5, fusion_y + 0.25, 'α·Spatial + (1-α)·Temporal',
            ha='center', va='center', fontsize=9)
    
    # ==================== WEATHER CROSS-ATTENTION ====================
    weather_y = 3.8
    weather_box = FancyBboxPatch((4, weather_y), 6, 0.7,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkblue', facecolor='#E8F0FF', linewidth=2)
    ax.add_patch(weather_box)
    ax.text(7, weather_y + 0.5, 'Weather Cross-Attention', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, weather_y + 0.2, 'Context-aware weather integration',
            ha='center', va='center', fontsize=9)
    
    # ==================== OUTPUT HEAD ====================
    output_y = 2.2
    output_box = FancyBboxPatch((4, output_y), 6, 1.2,
                                boxstyle="round,pad=0.1",
                                edgecolor='purple', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, output_y + 0.9, 'Gaussian Mixture Output', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, output_y + 0.55, '5 Mixture Components', ha='center', va='center', fontsize=10)
    ax.text(7, output_y + 0.25, '(μ₁, σ₁, π₁), ..., (μ₅, σ₅, π₅)',
            ha='center', va='center', fontsize=9)
    
    # ==================== FINAL OUTPUT ====================
    final_box = FancyBboxPatch((5, 0.5), 4, 0.7,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(final_box)
    ax.text(7, 0.95, 'Prediction', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, 0.65, 'Shape: [12, 62] (timesteps × nodes)',
            ha='center', va='center', fontsize=9)
    
    # ==================== ARROWS ====================
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to blocks
    ax.annotate('', xy=(2, block_y_start + 1.9), xytext=(2, 8),
                arrowprops=arrow_props)
    
    # Spatial to Fusion
    ax.annotate('', xy=(fusion_x + 0.5, fusion_y + 0.7), 
                xytext=(spatial_x + 2.5, spatial_y),
                arrowprops=arrow_props)
    
    # Temporal to Fusion
    ax.annotate('', xy=(fusion_x + 2.5, fusion_y + 0.7),
                xytext=(temporal_x + 1, temporal_y),
                arrowprops=arrow_props)
    
    # Fusion to Weather
    ax.annotate('', xy=(7, weather_y + 0.7), xytext=(7, fusion_y),
                arrowprops=arrow_props)
    
    # Weather to Output
    ax.annotate('', xy=(7, output_y + 1.2), xytext=(7, weather_y),
                arrowprops=arrow_props)
    
    # Output to Final
    ax.annotate('', xy=(7, 1.2), xytext=(7, output_y),
                arrowprops=arrow_props)
    
    # ==================== ANNOTATIONS ====================
    # Model specs
    specs_text = 'Model Specs:\n• Parameters: 680K\n• Hidden Dim: 96\n• Blocks: 3\n• Dropout: 0.2'
    ax.text(0.5, 3, specs_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    # Parallel processing highlight
    ax.annotate('', xy=(5.5, block_y_start + 1), xytext=(8.5, block_y_start + 1),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red', linestyle='--'))
    ax.text(7, block_y_start + 1.3, 'Parallel Processing', 
            ha='center', fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'fig11_stmgt_architecture')


def generate_fig12_attention_visualization():
    """Figure 12: Attention Mechanism Visualization"""
    print("Generating Figure 12: Attention Mechanisms...")
    
    fig = plt.figure(figsize=(14, 9))
    
    # Create 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ==================== SPATIAL ATTENTION (GATv2) ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('(a) Spatial Attention (GATv2)', fontsize=11, fontweight='bold', pad=8)
    
    # Draw graph nodes
    node_positions = {
        0: (5, 8),
        1: (3, 6),
        2: (7, 6),
        3: (5, 4),
        4: (2, 4),
        5: (8, 4)
    }
    
    # Target node (center)
    target_node = 0
    target_pos = node_positions[target_node]
    
    # Draw edges with attention weights (simulated)
    attention_weights = {1: 0.35, 2: 0.30, 3: 0.25, 4: 0.05, 5: 0.05}
    
    for neighbor, weight in attention_weights.items():
        neighbor_pos = node_positions[neighbor]
        # Draw arrow with thickness based on weight
        arrow = FancyArrowPatch(neighbor_pos, target_pos,
                               arrowstyle='->', mutation_scale=20,
                               lw=weight*10, color='steelblue', alpha=0.7)
        ax1.add_patch(arrow)
        
        # Add weight label
        mid_x = (neighbor_pos[0] + target_pos[0]) / 2
        mid_y = (neighbor_pos[1] + target_pos[1]) / 2
        ax1.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw nodes
    for node_id, pos in node_positions.items():
        if node_id == target_node:
            circle = Circle(pos, 0.4, color='red', ec='black', linewidth=2, zorder=10)
            label = f'Target\nNode {node_id}'
        else:
            circle = Circle(pos, 0.35, color='lightblue', ec='black', linewidth=1.5, zorder=10)
            label = f'N{node_id}'
        
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], label, ha='center', va='center', 
                fontsize=8, fontweight='bold' if node_id == target_node else 'normal')
    
    # Add formula
    formula_text = r'$\alpha_{ij} = \mathrm{softmax}(\mathbf{a}^T \cdot \mathrm{LeakyReLU}(W[h_i || h_j]))$'
    ax1.text(5, 1.5, formula_text, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.text(5, 0.5, 'Learns dynamic neighbor importance', ha='center', fontsize=8, style='italic')
    
    # ==================== TEMPORAL ATTENTION ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(b) Temporal Self-Attention', fontsize=11, fontweight='bold', pad=8)
    
    # Create attention heatmap (12 timesteps × 12 timesteps)
    np.random.seed(42)
    # Simulate attention pattern: recent timesteps have higher attention
    attention_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            # Higher attention to recent past and self
            if i >= j:
                attention_matrix[i, j] = np.exp(-0.3 * (i - j))
    
    # Normalize rows to sum to 1
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    im = ax2.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
    ax2.set_xlabel('Key Timestep', fontsize=10)
    ax2.set_ylabel('Query Timestep', fontsize=10)
    ax2.set_xticks(range(0, 12, 2))
    ax2.set_yticks(range(0, 12, 2))
    ax2.set_xticklabels([f't-{11-i}' for i in range(0, 12, 2)], fontsize=8)
    ax2.set_yticklabels([f't-{11-i}' for i in range(0, 12, 2)], fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=9)
    
    # Add annotation
    ax2.text(0.5, -0.15, 'Pattern: Recent timesteps receive higher attention',
            transform=ax2.transAxes, ha='center', fontsize=8, style='italic')
    
    # ==================== WEATHER CROSS-ATTENTION ====================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('(c) Weather Cross-Attention', fontsize=11, fontweight='bold', pad=8)
    
    # Traffic state box (Query)
    traffic_box = FancyBboxPatch((0.5, 6), 3, 2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkblue', facecolor='#E8F4F8', linewidth=2)
    ax3.add_patch(traffic_box)
    ax3.text(2, 7.5, 'Traffic State', ha='center', va='center', fontsize=9, fontweight='bold')
    ax3.text(2, 7, 'Query (Q)', ha='center', va='center', fontsize=8)
    ax3.text(2, 6.5, '12 × 62 × 96', ha='center', va='center', fontsize=7, style='italic')
    
    # Weather context box (Key/Value)
    weather_box = FancyBboxPatch((6.5, 6), 3, 2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkgreen', facecolor='#E8F8E8', linewidth=2)
    ax3.add_patch(weather_box)
    ax3.text(8, 7.5, 'Weather Context', ha='center', va='center', fontsize=9, fontweight='bold')
    ax3.text(8, 7, 'Key (K), Value (V)', ha='center', va='center', fontsize=8)
    ax3.text(8, 6.5, 'Temp, Wind, Precip', ha='center', va='center', fontsize=7, style='italic')
    
    # Cross-attention operation
    cross_box = FancyBboxPatch((3, 3.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='#F8E8FF', linewidth=2)
    ax3.add_patch(cross_box)
    ax3.text(5, 4.5, 'Cross-Attention', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(5, 4, r'$\mathrm{Attn}(Q_{traffic}, K_{weather}, V_{weather})$',
            ha='center', va='center', fontsize=8)
    
    # Output box
    output_box = FancyBboxPatch((3, 1), 4, 1.2,
                                boxstyle="round,pad=0.1",
                                edgecolor='darkorange', facecolor='#FFF4E5', linewidth=2)
    ax3.add_patch(output_box)
    ax3.text(5, 1.8, 'Weather-Aware Features', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax3.text(5, 1.4, 'Context-dependent weather effects', ha='center', va='center', fontsize=8)
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax3.annotate('', xy=(3.5, 4.8), xytext=(2, 6), arrowprops=arrow_props)
    ax3.annotate('', xy=(6.5, 4.8), xytext=(8, 6), arrowprops=arrow_props)
    ax3.annotate('', xy=(5, 2.2), xytext=(5, 3.5), arrowprops=arrow_props)
    
    # ==================== GATED FUSION ====================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('(d) Gated Fusion Mechanism', fontsize=11, fontweight='bold', pad=8)
    
    # Simulate gate values across timesteps (α values)
    timesteps = np.arange(12)
    # Higher α (spatial) during congestion, lower α (temporal) during free-flow
    gate_alpha = 0.5 + 0.3 * np.sin(timesteps * np.pi / 6)  # Oscillates between 0.2-0.8
    gate_beta = 1 - gate_alpha
    
    ax4.fill_between(timesteps, 0, gate_alpha, alpha=0.6, color='#FFE5E5', label='Spatial Weight (α)')
    ax4.fill_between(timesteps, gate_alpha, 1, alpha=0.6, color='#E5F5E5', label='Temporal Weight (β)')
    
    ax4.plot(timesteps, gate_alpha, 'o-', color='darkred', linewidth=2, markersize=6)
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax4.set_xlabel('Timestep', fontsize=10)
    ax4.set_ylabel('Gate Value', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_xticks(range(0, 12, 2))
    ax4.set_xticklabels([f't-{11-i}' for i in range(0, 12, 2)], fontsize=8)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    
    # Add formula
    formula = r'$h_{fused} = \alpha \odot h_{spatial} + (1-\alpha) \odot h_{temporal}$'
    ax4.text(0.5, -0.15, formula, transform=ax4.transAxes, 
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, 'fig12_attention_visualization')


def main():
    print(f"Output directory: {FIGURE_DIR}\n")
    
    generate_fig11_stmgt_architecture()
    generate_fig12_attention_visualization()
    
    print(f"\nArchitecture diagrams generated in: {FIGURE_DIR}")
    print("All 20 figures complete!")


if __name__ == '__main__':
    main()
