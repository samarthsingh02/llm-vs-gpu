"""
Visualization Index Generator
============================

Creates an HTML index page showcasing all generated visualizations.
"""

import os
from datetime import datetime

OUTPUT_DIR = 'visualizations'

# Visualization descriptions
VISUALIZATION_INFO = {
    'executive_dashboard.png': {
        'title': 'Executive Dashboard',
        'description': 'Comprehensive overview with key metrics, accuracy summary, and performance distribution',
        'category': 'Summary'
    },
    'performance_scatter.png': {
        'title': 'Performance Efficiency Landscape',
        'description': 'Scatter plot showing compute vs memory throughput with bottleneck classification',
        'category': 'Performance Analysis'
    },
    'prediction_accuracy_analysis.png': {
        'title': 'LLM Prediction Accuracy Analysis',
        'description': 'Detailed analysis of LLM prediction accuracy across different metrics and conditions',
        'category': 'LLM Analysis'
    },
    'cache_performance_analysis.png': {
        'title': 'Cache Performance Analysis',
        'description': 'Box plots showing cache hit rates and throughput by bottleneck type',
        'category': 'Performance Analysis'
    },
    'correlation_heatmap.png': {
        'title': 'Hardware Metrics Correlation Matrix',
        'description': 'Correlation analysis between all hardware performance metrics',
        'category': 'Statistical Analysis'
    },
    'feature_importance_analysis.png': {
        'title': 'Feature Importance Analysis',
        'description': 'Machine learning analysis showing which metrics are most predictive of bottleneck types',
        'category': 'Statistical Analysis'
    },
    '3d_performance_landscape.png': {
        'title': '3D Performance Landscape',
        'description': '3D visualization of kernel performance characteristics',
        'category': 'Performance Analysis'
    },
    '3d_performance_landscape.html': {
        'title': '3D Performance Landscape (Interactive)',
        'description': 'Interactive 3D plot for exploring kernel performance characteristics',
        'category': 'Performance Analysis'
    },
    'simple_radar_charts.png': {
        'title': 'Performance Radar Charts',
        'description': 'Multi-metric radar charts comparing key kernels across performance dimensions',
        'category': 'Performance Analysis'
    },
    'simple_distribution_plots.png': {
        'title': 'Performance Distribution Analysis',
        'description': 'Histograms and box plots showing distribution of performance metrics by bottleneck type',
        'category': 'Statistical Analysis'
    },
    'optimization_comparison.png': {
        'title': 'Optimization Impact Analysis',
        'description': 'Before/after comparison showing the impact of kernel optimizations',
        'category': 'Performance Analysis'
    },
    'prediction_confidence_analysis.png': {
        'title': 'Prediction Confidence Analysis',
        'description': 'Analysis of LLM prediction accuracy across different performance ranges and conditions',
        'category': 'LLM Analysis'
    }
}

def generate_html_index():
    """Generate HTML index page for all visualizations."""
    
    # Get list of actual files
    actual_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.html'))]
    
    # Group by category
    categories = {}
    for filename in actual_files:
        info = VISUALIZATION_INFO.get(filename, {
            'title': filename.replace('_', ' ').replace('.png', '').replace('.html', '').title(),
            'description': 'Generated visualization',
            'category': 'Other'
        })
        
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((filename, info))
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Performance Analysis - Visualization Gallery</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            text-align: center;
        }}
        
        .header h1 {{
            color: #333;
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.2em;
            margin: 0;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .stat {{
            background: rgba(255, 255, 255, 0.9);
            padding: 15px 25px;
            border-radius: 10px;
            margin: 5px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #4ECDC4;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .category {{
            margin-bottom: 40px;
        }}
        
        .category-header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
        }}
        
        .category-title {{
            color: #333;
            margin: 0;
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }}
        
        .viz-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        }}
        
        .viz-image {{
            width: 100%;
            height: 250px;
            object-fit: cover;
            object-position: center;
        }}
        
        .viz-content {{
            padding: 20px;
        }}
        
        .viz-title {{
            color: #333;
            margin: 0 0 10px 0;
            font-size: 1.3em;
            font-weight: bold;
        }}
        
        .viz-description {{
            color: #666;
            margin: 0 0 15px 0;
            line-height: 1.5;
        }}
        
        .viz-link {{
            display: inline-block;
            background: linear-gradient(45deg, #4ECDC4, #44A08D);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .viz-link:hover {{
            background: linear-gradient(45deg, #44A08D, #4ECDC4);
            transform: scale(1.05);
        }}
        
        .footer {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-top: 40px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
        }}
        
        .category-colors {{
            'Summary': '#FF6B6B',
            'Performance Analysis': '#4ECDC4',
            'LLM Analysis': '#45B7D1',
            'Statistical Analysis': '#96CEB4'
        }}
        
        @media (max-width: 768px) {{
            .visualization-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats {{
                flex-direction: column;
                align-items: center;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 GPU Performance Analysis</h1>
            <p>Comprehensive Visualization Gallery - LLM vs Hardware Ground Truth</p>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{len(actual_files)}</div>
                    <div class="stat-label">Total Visualizations</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{len(categories)}</div>
                    <div class="stat-label">Categories</div>
                </div>
                <div class="stat">
                    <div class="stat-number">63.6%</div>
                    <div class="stat-label">LLM Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-number">11</div>
                    <div class="stat-label">Kernels Analyzed</div>
                </div>
            </div>
        </div>
"""
    
    # Add categories
    category_order = ['Summary', 'Performance Analysis', 'LLM Analysis', 'Statistical Analysis', 'Other']
    
    for category in category_order:
        if category in categories:
            html_content += f"""
        <div class="category">
            <div class="category-header">
                <h2 class="category-title">{category}</h2>
            </div>
            
            <div class="visualization-grid">
"""
            
            for filename, info in categories[category]:
                # Determine if it's interactive
                is_interactive = filename.endswith('.html')
                link_text = "View Interactive" if is_interactive else "View Full Size"
                
                html_content += f"""
                <div class="viz-card">
                    {"" if is_interactive else f'<img src="{filename}" alt="{info["title"]}" class="viz-image">'}
                    <div class="viz-content">
                        <h3 class="viz-title">{info['title']}</h3>
                        <p class="viz-description">{info['description']}</p>
                        <a href="{filename}" target="_blank" class="viz-link">{link_text}</a>
                    </div>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
    
    # Footer
    html_content += f"""
        <div class="footer">
            <p><strong>Generated on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p><strong>Project:</strong> LLM vs GPU Performance Analysis | <strong>Team:</strong> GPU Performance Research</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    with open(f'{OUTPUT_DIR}/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_readme():
    """Generate README file for the visualizations directory."""
    
    actual_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.html'))]
    
    readme_content = f"""# GPU Performance Analysis - Visualization Gallery

This directory contains comprehensive visualizations analyzing GPU kernel performance and LLM prediction accuracy.

## 📊 Overview

- **Total Visualizations:** {len(actual_files)}
- **Analysis Date:** {datetime.now().strftime('%B %d, %Y')}
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
"""

    with open(f'{OUTPUT_DIR}/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    """Generate index and documentation."""
    print("📝 Generating visualization index and documentation...")
    
    generate_html_index()
    generate_readme()
    
    print(f"✅ Generated:")
    print(f"   📄 {OUTPUT_DIR}/index.html - Interactive gallery")
    print(f"   📋 {OUTPUT_DIR}/README.md - Documentation")
    print(f"\n🌐 Open index.html in your browser to explore all visualizations!")

if __name__ == "__main__":
    main()