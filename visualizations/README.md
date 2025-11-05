# GPU Performance Analysis - Visualization Gallery

This directory contains comprehensive visualizations analyzing GPU kernel performance and LLM prediction accuracy.

## 📊 Overview

- **Total Visualizations:** 13
- **Analysis Date:** November 05, 2025
- **LLM Overall Accuracy:** 63.6%
- **Kernels Analyzed:** 11

## 🎨 Available Visualizations

### Summary Dashboards
- **executive_dashboard.png** - Comprehensive overview with key metrics and performance distribution
- **index.html** - Interactive HTML gallery of all visualizations

### Performance Analysis
- **performance_scatter.png** - Compute vs Memory throughput landscape with bottleneck classification
- **cache_performance_analysis.png** - Cache hit rates and throughput analysis by bottleneck type
- **3d_performance_landscape.png/html** - 3D visualization of kernel performance characteristics
- **simple_radar_charts.png** - Multi-metric radar charts for key kernels
- **optimization_comparison.png** - Before/after optimization impact analysis

### LLM Analysis
- **prediction_accuracy_analysis.png** - Detailed LLM prediction accuracy breakdown
- **prediction_confidence_analysis.png** - Confidence analysis across performance ranges

### Statistical Analysis
- **correlation_heatmap.png** - Hardware metrics correlation matrix
- **feature_importance_analysis.png** - Machine learning feature importance analysis
- **simple_distribution_plots.png** - Performance metrics distribution by bottleneck type

## 🚀 Quick Start

1. Open `index.html` in your browser for an interactive gallery
2. Individual images can be viewed directly
3. The 3D landscape is available as both static PNG and interactive HTML

## 📈 Key Findings

### LLM Accuracy by Bottleneck Type:
- **Compute-Bound:** 100.0% accuracy (1 kernel)
- **Latency-Bound:** 100.0% accuracy (2 kernels)  
- **Memory-Bound:** 80.0% accuracy (5 kernels)
- **Mixed Workloads:** 0.0% accuracy (3 kernels)

### Performance Insights:
- Memory-bound kernels are most common (5/11 kernels)
- LLM struggles with mixed compute/memory workloads
- Clear bottleneck types (compute-only, latency-only) are predicted perfectly
- Hardware metrics show clear separation between bottleneck types

## 🔧 Technical Details

All visualizations are generated using:
- **matplotlib** - Static plots with custom styling
- **seaborn** - Statistical visualizations
- **plotly** - Interactive 3D plots
- **scikit-learn** - Machine learning analysis

Plot styling uses:
- Professional color palettes
- Consistent typography (Arial font family)
- High-resolution output (200-300 DPI)
- Accessible color schemes
