"""
Advanced Visualization Suite for LLM vs GPU Performance Analysis
================================================================

This script generates comprehensive visualizations for analyzing GPU kernel performance
and LLM prediction accuracy. All plots are saved in the 'visualizations/' directory.

Author: GPU Performance Analysis Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURATION & SETUP
# ========================================================================================

# Set up beautiful matplotlib styling
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
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

# Beautiful color palettes
BOTTLENECK_COLORS = {
    'Compute-Bound': '#FF6B6B',          # Coral Red
    'Memory-Bound': '#4ECDC4',           # Teal
    'Latency-Bound': '#45B7D1',          # Sky Blue
    'MIXED(Compute/Memory)': '#96CEB4',   # Mint Green
    'Likely COMPUTE-BOUND': '#FFEAA7',    # Light Yellow
    'Likely MEMORY-BOUND': '#DDA0DD',     # Plum
    'MIXED/OTHER': '#95A5A6'             # Gray
}

GRADIENT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

# File paths
OUTPUT_DIR = 'visualizations'
GROUND_TRUTH_PATH = 'data/derived_ground_truth.csv'
LLM_PREDICTIONS_PATH = 'data/llm_predictions.csv'

print("🎨 Advanced GPU Performance Visualization Suite")
print("=" * 60)

# ========================================================================================
# DATA LOADING & PREPROCESSING
# ========================================================================================

def load_and_prepare_data():
    """Load and prepare data for visualization."""
    print("📊 Loading data...")
    
    # Load ground truth data
    df_gt = pd.read_csv(GROUND_TRUTH_PATH, index_col=0)
    
    # Load LLM predictions
    df_llm = pd.read_csv(LLM_PREDICTIONS_PATH)
    
    # Clean LLM kernel names
    def clean_llm_kernel_name(name):
        import re
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
    
    print(f"✅ Data loaded: {len(merged_df)} kernels with complete data")
    return df_gt, merged_df

# ========================================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================================

def create_performance_scatter(merged_df):
    """1. Performance Efficiency Scatter Plot"""
    print("🔵 Creating Performance Efficiency Scatter Plot...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with bottleneck classification
    for bottleneck, color in BOTTLENECK_COLORS.items():
        mask = merged_df['ground_truth_label'] == bottleneck
        if mask.any():
            ax.scatter(merged_df.loc[mask, 'Compute (SM) Throughput'], 
                      merged_df.loc[mask, 'DRAM Throughput'],
                      c=color, label=bottleneck, s=120, alpha=0.7, 
                      edgecolors='white', linewidth=2)
    
    # Add quadrant lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add region labels
    ax.text(75, 25, 'Compute\nDominant', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    ax.text(25, 75, 'Memory\nDominant', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    ax.text(75, 75, 'Mixed\nWorkload', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    ax.text(25, 25, 'Latency\nBound', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlabel('Compute (SM) Throughput (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('DRAM Throughput (%)', fontsize=14, fontweight='bold')
    ax.set_title('GPU Performance Efficiency Landscape\nCompute vs Memory Throughput by Bottleneck Type', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_accuracy_analysis(merged_df):
    """2. LLM Prediction Accuracy Analysis"""
    print("🧠 Creating LLM Prediction Accuracy Analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy by bottleneck type
    accuracy_by_type = merged_df.groupby('ground_truth_label')['is_correct'].agg(['mean', 'count'])
    accuracy_by_type['accuracy_pct'] = accuracy_by_type['mean'] * 100
    
    bars1 = ax1.bar(range(len(accuracy_by_type)), accuracy_by_type['accuracy_pct'], 
                    color=[BOTTLENECK_COLORS.get(label, '#95A5A6') for label in accuracy_by_type.index],
                    edgecolor='white', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('LLM Accuracy by Bottleneck Type', fontweight='bold', pad=15)
    ax1.set_xticks(range(len(accuracy_by_type)))
    ax1.set_xticklabels(accuracy_by_type.index, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars1, accuracy_by_type['accuracy_pct']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Error distribution by SM throughput
    correct_sm = merged_df[merged_df['is_correct']]['Compute (SM) Throughput']
    incorrect_sm = merged_df[~merged_df['is_correct']]['Compute (SM) Throughput']
    
    ax2.hist(correct_sm, bins=10, alpha=0.7, label='Correct Predictions', 
             color='#4ECDC4', edgecolor='white', linewidth=1)
    ax2.hist(incorrect_sm, bins=10, alpha=0.7, label='Incorrect Predictions', 
             color='#FF6B6B', edgecolor='white', linewidth=1)
    ax2.set_xlabel('Compute (SM) Throughput (%)', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Prediction Accuracy vs SM Throughput', fontweight='bold', pad=15)
    ax2.legend()
    
    # Error distribution by DRAM throughput
    correct_dram = merged_df[merged_df['is_correct']]['DRAM Throughput']
    incorrect_dram = merged_df[~merged_df['is_correct']]['DRAM Throughput']
    
    ax3.hist(correct_dram, bins=10, alpha=0.7, label='Correct Predictions', 
             color='#4ECDC4', edgecolor='white', linewidth=1)
    ax3.hist(incorrect_dram, bins=10, alpha=0.7, label='Incorrect Predictions', 
             color='#FF6B6B', edgecolor='white', linewidth=1)
    ax3.set_xlabel('DRAM Throughput (%)', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Prediction Accuracy vs DRAM Throughput', fontweight='bold', pad=15)
    ax3.legend()
    
    # Confusion matrix heatmap
    from sklearn.metrics import confusion_matrix
    labels = merged_df['ground_truth_label'].unique()
    cm = confusion_matrix(merged_df['ground_truth_label'], merged_df['llm_label'], labels=labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=[label.replace(' ', '\n') for label in labels],
                yticklabels=[label.replace(' ', '\n') for label in labels],
                cbar_kws={'shrink': 0.8})
    ax4.set_title('Prediction Confusion Matrix', fontweight='bold', pad=15)
    ax4.set_xlabel('Predicted Label', fontweight='bold')
    ax4.set_ylabel('Actual Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/prediction_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cache_performance_analysis(merged_df):
    """3. Cache Performance Analysis"""
    print("💾 Creating Cache Performance Analysis...")
    
    # Check if cache metrics are available
    cache_metrics = ['L1/TEX Hit Rate', 'L2 Hit Rate', 'L1/TEX Cache Throughput', 'L2 Cache Throughput']
    available_metrics = [metric for metric in cache_metrics if metric in merged_df.columns]
    
    if not available_metrics:
        print("⚠️  Cache metrics not available, skipping cache analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        
        # Box plot by bottleneck type
        bottleneck_types = merged_df['ground_truth_label'].unique()
        data_by_type = [merged_df[merged_df['ground_truth_label'] == bt][metric].dropna() 
                       for bt in bottleneck_types]
        
        bp = ax.boxplot(data_by_type, labels=[bt.replace(' ', '\n') for bt in bottleneck_types],
                       patch_artist=True, showfliers=True)
        
        # Color the boxes
        for patch, bottleneck in zip(bp['boxes'], bottleneck_types):
            patch.set_facecolor(BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'))
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric} by Bottleneck Type', fontweight='bold', pad=15)
        ax.set_ylabel(metric, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cache_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(merged_df):
    """4. Hardware Metrics Correlation Matrix"""
    print("🔗 Creating Correlation Heatmap...")
    
    # Select numeric columns for correlation
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    # Remove boolean columns
    numeric_cols = [col for col in numeric_cols if not merged_df[col].dtype == bool]
    
    corr_matrix = merged_df[numeric_cols].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, annot=True, fmt='.2f', 
                cbar_kws={"shrink": .8}, ax=ax,
                annot_kws={'size': 10})
    
    ax.set_title('Hardware Metrics Correlation Matrix\nIdentifying Key Performance Relationships', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_radar_charts(merged_df):
    """5. Multi-Metric Radar Charts"""
    print("📡 Creating Performance Radar Charts...")
    
    # Select key metrics for radar chart
    radar_metrics = ['Compute (SM) Throughput', 'DRAM Throughput', 'SM Busy']
    
    # Add cache metrics if available
    if 'L1/TEX Hit Rate' in merged_df.columns:
        radar_metrics.append('L1/TEX Hit Rate')
    if 'L2 Hit Rate' in merged_df.columns:
        radar_metrics.append('L2 Hit Rate')
    
    # Limit to available metrics
    radar_metrics = [metric for metric in radar_metrics if metric in merged_df.columns]
    
    if len(radar_metrics) < 3:
        print("⚠️  Not enough metrics for radar chart, skipping")
        return
    
    # Select top 6 kernels for visualization
    kernels_to_plot = merged_df.index[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for i, kernel in enumerate(kernels_to_plot):
        ax = axes[i]
        
        # Get values for this kernel
        values = merged_df.loc[kernel, radar_metrics].values
        values = np.nan_to_num(values, nan=0)  # Replace NaN with 0
        
        # Number of variables
        N = len(radar_metrics)
        
        # Angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values = np.concatenate((values, [values[0]]))  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=GRADIENT_COLORS[i % len(GRADIENT_COLORS)])
        ax.fill(angles, values, alpha=0.25, color=GRADIENT_COLORS[i % len(GRADIENT_COLORS)])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace(' ', '\n') for metric in radar_metrics], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'{kernel}\n({merged_df.loc[kernel, "ground_truth_label"]})', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_analysis(merged_df):
    """6. Performance Metrics Distribution Analysis"""
    print("📊 Creating Distribution Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution of SM Throughput by bottleneck type
    bottleneck_types = merged_df['ground_truth_label'].unique()
    
    for i, bottleneck in enumerate(bottleneck_types):
        if i >= 4:  # Limit to 4 for subplot layout
            break
        ax = axes[i // 2, i % 2]
        
        data = merged_df[merged_df['ground_truth_label'] == bottleneck]['Compute (SM) Throughput']
        
        # Histogram
        ax.hist(data, bins=15, alpha=0.7, color=BOTTLENECK_COLORS.get(bottleneck, '#95A5A6'),
                edgecolor='white', linewidth=1)
        
        # Add vertical line for mean
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.1f}%')
        
        ax.set_xlabel('Compute (SM) Throughput (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{bottleneck}\nSM Throughput Distribution', fontweight='bold', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimization_comparison(merged_df):
    """7. Before/After Optimization Comparison"""
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
        print("⚠️  No optimization pairs found, skipping comparison")
        return
    
    fig, axes = plt.subplots(len(available_pairs), 2, figsize=(16, 6 * len(available_pairs)))
    if len(available_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    metrics = ['Compute (SM) Throughput', 'DRAM Throughput']
    
    for pair_idx, (before, after) in enumerate(available_pairs):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[pair_idx, metric_idx]
            
            before_val = merged_df.loc[before, metric]
            after_val = merged_df.loc[after, metric]
            
            bars = ax.bar(['Before\n(Naive)', 'After\n(Optimized)'], [before_val, after_val],
                         color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels
            for bar, val in zip(bars, [before_val, after_val]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Calculate improvement
            improvement = ((after_val - before_val) / before_val) * 100
            ax.text(0.5, max(before_val, after_val) * 0.8, 
                   f'Improvement:\n{improvement:+.1f}%', 
                   ha='center', va='center', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            
            ax.set_ylabel(f'{metric} (%)', fontweight='bold')
            ax.set_title(f'{before.replace("_", " ").title()} vs {after.replace("_", " ").title()}\n{metric}', 
                        fontweight='bold', pad=15)
            ax.set_ylim(0, max(before_val, after_val) * 1.2)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_performance_landscape(merged_df):
    """8. 3D Performance Landscape (Interactive)"""
    print("🌄 Creating 3D Performance Landscape...")
    
    # Check if we have enough metrics for 3D plot
    required_metrics = ['Compute (SM) Throughput', 'DRAM Throughput']
    z_metric = 'SM Busy' if 'SM Busy' in merged_df.columns else None
    
    if not all(metric in merged_df.columns for metric in required_metrics):
        print("⚠️  Required metrics not available for 3D plot")
        return
    
    if z_metric is None:
        print("⚠️  Z-axis metric not available, using L1/TEX Hit Rate or default")
        if 'L1/TEX Hit Rate' in merged_df.columns:
            z_metric = 'L1/TEX Hit Rate'
        else:
            # Create a synthetic z-metric
            z_metric = 'Performance_Score'
            merged_df[z_metric] = (merged_df['Compute (SM) Throughput'] + merged_df['DRAM Throughput']) / 2
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=merged_df['Compute (SM) Throughput'],
        y=merged_df['DRAM Throughput'], 
        z=merged_df[z_metric],
        mode='markers+text',
        marker=dict(
            size=12,
            color=[BOTTLENECK_COLORS.get(label, '#95A5A6') for label in merged_df['ground_truth_label']],
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=merged_df.index,
        textposition="top center",
        textfont=dict(size=10),
        name='Kernels'
    ))
    
    fig.update_layout(
        title=dict(
            text='3D GPU Performance Landscape<br><sub>Interactive Exploration of Kernel Characteristics</sub>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Compute (SM) Throughput (%)',
            yaxis_title='DRAM Throughput (%)',
            zaxis_title=f'{z_metric} (%)',
            bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        width=1000,
        height=800,
        font=dict(family="Arial", size=12)
    )
    
    # Save as HTML for interactivity
    fig.write_html(f'{OUTPUT_DIR}/3d_performance_landscape.html')
    
    # Also save as static PNG
    fig.write_image(f'{OUTPUT_DIR}/3d_performance_landscape.png', width=1000, height=800)

def create_feature_importance_analysis(merged_df):
    """9. Feature Importance Analysis"""
    print("🎯 Creating Feature Importance Analysis...")
    
    # Prepare data for machine learning
    feature_cols = merged_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col not in ['is_correct'] and not merged_df[col].dtype == bool]
    
    X = merged_df[feature_cols].fillna(0)
    y = merged_df['ground_truth_label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))),
                   edgecolor='white', linewidth=1)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels([feat.replace(' ', '\n') for feat in importance_df['feature']], fontsize=11)
    ax.set_xlabel('Feature Importance Score', fontweight='bold', fontsize=14)
    ax.set_title('Hardware Metrics Feature Importance\nfor Bottleneck Classification', 
                 fontweight='bold', fontsize=16, pad=20)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(merged_df):
    """10. Executive Summary Dashboard"""
    print("📋 Creating Executive Summary Dashboard...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Overall accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accuracy = merged_df['is_correct'].mean() * 100
    ax1.pie([accuracy, 100-accuracy], labels=['Correct', 'Incorrect'], 
            colors=['#4ECDC4', '#FF6B6B'], autopct='%1.1f%%', startangle=90,
            textprops={'fontweight': 'bold', 'fontsize': 12})
    ax1.set_title(f'Overall LLM Accuracy\n{accuracy:.1f}%', fontweight='bold', fontsize=14)
    
    # Bottleneck distribution
    ax2 = fig.add_subplot(gs[0, 1:3])
    bottleneck_counts = merged_df['ground_truth_label'].value_counts()
    bars = ax2.bar(range(len(bottleneck_counts)), bottleneck_counts.values,
                   color=[BOTTLENECK_COLORS.get(label, '#95A5A6') for label in bottleneck_counts.index],
                   edgecolor='white', linewidth=2)
    ax2.set_xticks(range(len(bottleneck_counts)))
    ax2.set_xticklabels([label.replace(' ', '\n') for label in bottleneck_counts.index], 
                       rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Number of Kernels', fontweight='bold')
    ax2.set_title('Bottleneck Type Distribution', fontweight='bold', fontsize=14)
    
    # Performance metrics summary
    ax3 = fig.add_subplot(gs[0, 3])
    metrics_summary = merged_df[['Compute (SM) Throughput', 'DRAM Throughput']].describe()
    ax3.table(cellText=metrics_summary.round(1).values,
              rowLabels=metrics_summary.index,
              colLabels=[col.replace(' ', '\n') for col in metrics_summary.columns],
              cellLoc='center', loc='center')
    ax3.axis('off')
    ax3.set_title('Performance Metrics\nSummary Statistics', fontweight='bold', fontsize=14)
    
    # Top performing kernels
    ax4 = fig.add_subplot(gs[1, :2])
    top_compute = merged_df.nlargest(5, 'Compute (SM) Throughput')
    bars = ax4.barh(range(len(top_compute)), top_compute['Compute (SM) Throughput'],
                    color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=1)
    ax4.set_yticks(range(len(top_compute)))
    ax4.set_yticklabels(top_compute.index, fontsize=11)
    ax4.set_xlabel('Compute Throughput (%)', fontweight='bold')
    ax4.set_title('Top 5 Compute-Intensive Kernels', fontweight='bold', fontsize=14)
    
    # Top memory kernels
    ax5 = fig.add_subplot(gs[1, 2:])
    top_memory = merged_df.nlargest(5, 'DRAM Throughput')
    bars = ax5.barh(range(len(top_memory)), top_memory['DRAM Throughput'],
                    color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=1)
    ax5.set_yticks(range(len(top_memory)))
    ax5.set_yticklabels(top_memory.index, fontsize=11)
    ax5.set_xlabel('DRAM Throughput (%)', fontweight='bold')
    ax5.set_title('Top 5 Memory-Intensive Kernels', fontweight='bold', fontsize=14)
    
    # Prediction accuracy heatmap
    ax6 = fig.add_subplot(gs[2:, :2])
    accuracy_matrix = merged_df.groupby(['ground_truth_label', 'llm_label']).size().unstack(fill_value=0)
    sns.heatmap(accuracy_matrix, annot=True, fmt='d', cmap='Blues', ax=ax6,
                cbar_kws={'shrink': 0.8})
    ax6.set_title('Detailed Prediction Matrix', fontweight='bold', fontsize=14)
    ax6.set_xlabel('LLM Predicted', fontweight='bold')
    ax6.set_ylabel('Ground Truth', fontweight='bold')
    
    # Performance scatter
    ax7 = fig.add_subplot(gs[2:, 2:])
    for bottleneck, color in BOTTLENECK_COLORS.items():
        mask = merged_df['ground_truth_label'] == bottleneck
        if mask.any():
            ax7.scatter(merged_df.loc[mask, 'Compute (SM) Throughput'], 
                       merged_df.loc[mask, 'DRAM Throughput'],
                       c=color, label=bottleneck, s=100, alpha=0.7, 
                       edgecolors='white', linewidth=2)
    
    ax7.set_xlabel('Compute (SM) Throughput (%)', fontweight='bold')
    ax7.set_ylabel('DRAM Throughput (%)', fontweight='bold')
    ax7.set_title('Performance Landscape Overview', fontweight='bold', fontsize=14)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('GPU Performance Analysis Executive Dashboard', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig(f'{OUTPUT_DIR}/executive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """Main execution function."""
    print("🚀 Starting comprehensive visualization generation...")
    
    # Load data
    df_gt, merged_df = load_and_prepare_data()
    
    # Create all visualizations
    visualization_functions = [
        create_performance_scatter,
        create_prediction_accuracy_analysis,
        create_cache_performance_analysis,
        create_correlation_heatmap,
        create_performance_radar_charts,
        create_distribution_analysis,
        create_optimization_comparison,
        create_3d_performance_landscape,
        create_feature_importance_analysis,
        create_summary_dashboard
    ]
    
    total_plots = len(visualization_functions)
    
    for i, viz_func in enumerate(visualization_functions, 1):
        try:
            viz_func(merged_df)
            print(f"✅ [{i}/{total_plots}] {viz_func.__name__} completed")
        except Exception as e:
            print(f"❌ [{i}/{total_plots}] {viz_func.__name__} failed: {str(e)}")
    
    print("\n" + "="*60)
    print("🎉 VISUALIZATION SUITE COMPLETE!")
    print(f"📁 All plots saved to: {OUTPUT_DIR}/")
    print("="*60)
    
    # Generate summary report
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Total kernels analyzed: {len(merged_df)}")
    print(f"Overall LLM accuracy: {merged_df['is_correct'].mean()*100:.1f}%")
    print(f"Bottleneck types found: {len(merged_df['ground_truth_label'].unique())}")
    
    # Accuracy by type
    print("\n🎯 ACCURACY BY BOTTLENECK TYPE:")
    accuracy_by_type = merged_df.groupby('ground_truth_label')['is_correct'].agg(['mean', 'count'])
    for bottleneck, row in accuracy_by_type.iterrows():
        print(f"  {bottleneck}: {row['mean']*100:.1f}% ({row['count']} kernels)")

if __name__ == "__main__":
    main()