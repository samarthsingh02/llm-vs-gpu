"""
Simple Visualization Additions - Memory-Efficient Versions
==========================================================

This script creates the remaining visualizations with memory-efficient approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful matplotlib styling
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Color palette
BOTTLENECK_COLORS = {
    'Compute-Bound': '#FF6B6B',
    'Memory-Bound': '#4ECDC4',
    'Latency-Bound': '#45B7D1',
    'MIXED(Compute/Memory)': '#96CEB4',
    'Likely COMPUTE-BOUND': '#FFEAA7',
    'Likely MEMORY-BOUND': '#DDA0DD',
    'MIXED/OTHER': '#95A5A6'
}

OUTPUT_DIR = 'visualizations'

def load_data():
    """Load the processed data."""
    df_gt = pd.read_csv('data/derived_ground_truth.csv', index_col=0)
    df_llm = pd.read_csv('data/llm_predictions.csv')
    
    # Clean LLM kernel names
    import re
    def clean_llm_kernel_name(name):
        cleaned = re.sub(r'^\d+_', '', str(name))
        return cleaned.replace('.cu', '')
    
    if 'kernel' in df_llm.columns:
        df_llm = df_llm.rename(columns={'kernel': 'kernel_name'})
    
    df_llm['kernel_name'] = df_llm['kernel_name'].apply(clean_llm_kernel_name)
    df_llm['label'] = df_llm['label'].str.title()
    df_llm = df_llm.set_index('kernel_name')
    
    # Merge datasets
    merged_df = df_gt.join(df_llm, how='inner')
    merged_df = merged_df.rename(columns={'label': 'llm_label', 'rationale': 'llm_rationale'})
    merged_df['is_correct'] = (merged_df['ground_truth_label'] == merged_df['llm_label'])
    
    return merged_df

def create_simple_radar_chart(merged_df):
    """Create a simple radar chart for top kernels."""
    print("📡 Creating Simple Radar Chart...")
    
    # Simple approach - just 3 kernels, 4 metrics
    radar_metrics = ['Compute (SM) Throughput', 'DRAM Throughput', 'SM Busy']
    
    # Add one more metric if available
    if 'L1/TEX Hit Rate' in merged_df.columns:
        radar_metrics.append('L1/TEX Hit Rate')
    
    # Select 3 representative kernels
    kernels_to_plot = ['saxpy', 'naive_matmul', 'tiled_matmul']
    available_kernels = [k for k in kernels_to_plot if k in merged_df.index]
    
    if len(available_kernels) == 0:
        print("⚠️  No target kernels available for radar chart")
        return
    
    fig, axes = plt.subplots(1, len(available_kernels), figsize=(5 * len(available_kernels), 5), 
                            subplot_kw=dict(projection='polar'))
    if len(available_kernels) == 1:
        axes = [axes]
    
    for i, kernel in enumerate(available_kernels):
        ax = axes[i]
        
        # Get values for this kernel
        values = []
        for metric in radar_metrics:
            if metric in merged_df.columns:
                val = merged_df.loc[kernel, metric]
                values.append(val if not pd.isna(val) else 0)
            else:
                values.append(0)
        
        # Number of variables
        N = len(radar_metrics)
        
        # Angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values = values + [values[0]]  # Complete the circle
        
        # Plot
        color = BOTTLENECK_COLORS.get(merged_df.loc[kernel, 'ground_truth_label'], '#95A5A6')
        ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.25, color=color)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace(' ', '\n') for metric in radar_metrics], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'{kernel.replace("_", " ").title()}\n({merged_df.loc[kernel, "ground_truth_label"]})', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/simple_radar_charts.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()  # Clear figure

def create_simple_distribution_plots(merged_df):
    """Create simple distribution plots."""
    print("📊 Creating Simple Distribution Plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. SM Throughput by bottleneck type
    bottleneck_types = merged_df['ground_truth_label'].unique()[:4]  # Limit to 4 types
    
    for bottleneck in bottleneck_types:
        data = merged_df[merged_df['ground_truth_label'] == bottleneck]['Compute (SM) Throughput']
        ax1.hist(data, bins=8, alpha=0.6, 
                label=bottleneck.replace(' ', '\n'), 
                color=BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'),
                edgecolor='white', linewidth=1)
    
    ax1.set_xlabel('Compute (SM) Throughput (%)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('SM Throughput Distribution\nby Bottleneck Type', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. DRAM Throughput distribution
    for bottleneck in bottleneck_types:
        data = merged_df[merged_df['ground_truth_label'] == bottleneck]['DRAM Throughput']
        ax2.hist(data, bins=8, alpha=0.6, 
                label=bottleneck.replace(' ', '\n'), 
                color=BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'),
                edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('DRAM Throughput (%)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('DRAM Throughput Distribution\nby Bottleneck Type', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot for SM Throughput
    data_for_box = [merged_df[merged_df['ground_truth_label'] == bt]['Compute (SM) Throughput'].values 
                    for bt in bottleneck_types]
    
    bp1 = ax3.boxplot(data_for_box, labels=[bt.replace(' ', '\n') for bt in bottleneck_types],
                     patch_artist=True)
    
    for patch, bottleneck in zip(bp1['boxes'], bottleneck_types):
        patch.set_facecolor(BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'))
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Compute (SM) Throughput (%)', fontweight='bold')
    ax3.set_title('SM Throughput Box Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot for DRAM Throughput
    data_for_box = [merged_df[merged_df['ground_truth_label'] == bt]['DRAM Throughput'].values 
                    for bt in bottleneck_types]
    
    bp2 = ax4.boxplot(data_for_box, labels=[bt.replace(' ', '\n') for bt in bottleneck_types],
                     patch_artist=True)
    
    for patch, bottleneck in zip(bp2['boxes'], bottleneck_types):
        patch.set_facecolor(BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'))
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('DRAM Throughput (%)', fontweight='bold')
    ax4.set_title('DRAM Throughput Box Plot', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/simple_distribution_plots.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()

def create_optimization_comparison(merged_df):
    """Create optimization comparison chart."""
    print("🚀 Creating Optimization Comparison...")
    
    # Define optimization pairs
    optimization_pairs = [
        ('naive_matmul', 'tiled_matmul'),
        ('naive_transpose', 'tiled_transpose')
    ]
    
    available_pairs = []
    for before, after in optimization_pairs:
        if before in merged_df.index and after in merged_df.index:
            available_pairs.append((before, after))
    
    if not available_pairs:
        print("⚠️  No optimization pairs found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = ['Compute (SM) Throughput', 'DRAM Throughput']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        x_pos = 0
        bar_width = 0.35
        
        for pair_idx, (before, after) in enumerate(available_pairs):
            before_val = merged_df.loc[before, metric]
            after_val = merged_df.loc[after, metric]
            
            # Create grouped bars
            x_before = x_pos
            x_after = x_pos + bar_width
            
            ax.bar(x_before, before_val, bar_width, label='Before' if pair_idx == 0 else "", 
                   color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=1)
            ax.bar(x_after, after_val, bar_width, label='After' if pair_idx == 0 else "", 
                   color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add improvement annotation
            improvement = ((after_val - before_val) / before_val) * 100
            ax.annotate(f'{improvement:+.1f}%', 
                       xy=(x_pos + bar_width/2, max(before_val, after_val) + 5),
                       ha='center', va='bottom', fontweight='bold', fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            
            # Add pair labels
            ax.text(x_pos + bar_width/2, -10, 
                   f'{before.replace("_", " ")}\nvs\n{after.replace("_", " ")}',
                   ha='center', va='top', fontsize=10, fontweight='bold')
            
            x_pos += 1.5  # Move to next pair
        
        ax.set_ylabel(f'{metric} (%)', fontweight='bold')
        ax.set_title(f'Optimization Impact on {metric}', fontweight='bold', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])  # Remove x-axis ticks since we have custom labels
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimization_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()

def create_prediction_confidence_analysis(merged_df):
    """Create prediction confidence analysis."""
    print("🎯 Creating Prediction Confidence Analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy vs SM Throughput ranges
    sm_ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    accuracies = []
    counts = []
    
    for low, high in sm_ranges:
        mask = (merged_df['Compute (SM) Throughput'] >= low) & (merged_df['Compute (SM) Throughput'] < high)
        if mask.sum() > 0:
            acc = merged_df[mask]['is_correct'].mean() * 100
            count = mask.sum()
        else:
            acc = 0
            count = 0
        accuracies.append(acc)
        counts.append(count)
    
    bars1 = ax1.bar(range(len(sm_ranges)), accuracies, 
                   color=['#FF6B6B', '#FFA07A', '#20B2AA', '#4ECDC4'],
                   edgecolor='white', linewidth=2)
    ax1.set_xticks(range(len(sm_ranges)))
    ax1.set_xticklabels([f'{low}-{high}%' for low, high in sm_ranges])
    ax1.set_ylabel('LLM Accuracy (%)', fontweight='bold')
    ax1.set_xlabel('SM Throughput Range', fontweight='bold')
    ax1.set_title('LLM Accuracy vs SM Throughput Ranges', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add count labels
    for bar, acc, count in zip(bars1, accuracies, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Accuracy vs DRAM Throughput ranges
    dram_ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    accuracies = []
    counts = []
    
    for low, high in dram_ranges:
        mask = (merged_df['DRAM Throughput'] >= low) & (merged_df['DRAM Throughput'] < high)
        if mask.sum() > 0:
            acc = merged_df[mask]['is_correct'].mean() * 100
            count = mask.sum()
        else:
            acc = 0
            count = 0
        accuracies.append(acc)
        counts.append(count)
    
    bars2 = ax2.bar(range(len(dram_ranges)), accuracies, 
                   color=['#45B7D1', '#87CEEB', '#98FB98', '#96CEB4'],
                   edgecolor='white', linewidth=2)
    ax2.set_xticks(range(len(dram_ranges)))
    ax2.set_xticklabels([f'{low}-{high}%' for low, high in dram_ranges])
    ax2.set_ylabel('LLM Accuracy (%)', fontweight='bold')
    ax2.set_xlabel('DRAM Throughput Range', fontweight='bold')
    ax2.set_title('LLM Accuracy vs DRAM Throughput Ranges', fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add count labels
    for bar, acc, count in zip(bars2, accuracies, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Kernel complexity vs accuracy (using kernel name length as proxy for complexity)
    merged_df['kernel_complexity'] = merged_df.index.str.len()
    
    ax3.scatter(merged_df['kernel_complexity'], merged_df['is_correct'].astype(int), 
               c=[BOTTLENECK_COLORS.get(label, '#95A5A6') for label in merged_df['ground_truth_label']],
               s=100, alpha=0.7, edgecolors='white', linewidth=2)
    
    ax3.set_xlabel('Kernel Name Length (Complexity Proxy)', fontweight='bold')
    ax3.set_ylabel('Prediction Correct (1=Yes, 0=No)', fontweight='bold')
    ax3.set_title('Prediction Accuracy vs Kernel Complexity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance balance vs accuracy
    merged_df['performance_balance'] = abs(merged_df['Compute (SM) Throughput'] - merged_df['DRAM Throughput'])
    
    correct_balance = merged_df[merged_df['is_correct']]['performance_balance']
    incorrect_balance = merged_df[~merged_df['is_correct']]['performance_balance']
    
    ax4.hist(correct_balance, bins=6, alpha=0.7, label='Correct Predictions', 
             color='#4ECDC4', edgecolor='white', linewidth=1)
    ax4.hist(incorrect_balance, bins=6, alpha=0.7, label='Incorrect Predictions', 
             color='#FF6B6B', edgecolor='white', linewidth=1)
    
    ax4.set_xlabel('Performance Balance\n|SM Throughput - DRAM Throughput|', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Prediction Accuracy vs Performance Balance', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/prediction_confidence_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()

def main():
    """Run all remaining visualizations."""
    print("🎨 Creating Additional Beautiful Visualizations")
    print("=" * 50)
    
    merged_df = load_data()
    
    # Create the remaining plots
    create_simple_radar_chart(merged_df)
    create_simple_distribution_plots(merged_df)
    create_optimization_comparison(merged_df)
    create_prediction_confidence_analysis(merged_df)
    
    print("\n✅ All additional visualizations completed!")
    print(f"📁 Saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()