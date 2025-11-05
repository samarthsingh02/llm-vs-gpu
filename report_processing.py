import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set matplotlib and seaborn styling for prettier plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.autolayout'] = True
sns.set_palette("husl")

# --- Configuration -- -
NCU_CSV_PATH = 'ncu_report_details.csv'
OUTPUT_CSV_PATH = 'data/derived_ground_truth.csv'
LLM_CSV_PATH = 'data/llm_predictions.csv'
MISMATCH_CSV_PATH = 'data/mismatched_predictions.csv'

# Define the metrics we absolutely need for classification
# Match the human-readable names present in ncu_report_details.csv
# Example rows show: "Compute (SM) Throughput" and "DRAM Throughput" (both in %)
REQUIRED_METRIC_NAMES = [
    'Compute (SM) Throughput',
    'DRAM Throughput'
]

# Optional metrics (best-effort if present in the CSV)
OPTIONAL_METRIC_NAMES = [
    'SM Busy',
    'Mem Pipes Busy',
    'L1/TEX Hit Rate',
    'L2 Hit Rate',
    'L1/TEX Cache Throughput',
    'L2 Cache Throughput',
    'Warp Cycles Per Issued Instruction',
]

KERNEL_NAME_COL = 'Kernel Name' # Column name in the NCU CSV

# --- Phase 1: Ground Truth Extraction & Preparation ---

print("--- Phase 1: Processing Nsight Compute Data ---")

# Load the detailed report - try specifying utf-16 encoding
try:
    # First, read just a few lines to find the header
    with open(NCU_CSV_PATH, 'r', encoding='utf-16') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if i > 50:  # Only read first 50 lines to find header
                break
    
    skiprows = 0
    for i, line in enumerate(lines):
        if KERNEL_NAME_COL in line:
            skiprows = i
            break
    
    print(f"Found header at row {skiprows}")
    
    # Now read the CSV in chunks to handle large files
    chunk_list = []
    try:
        chunk_iter = pd.read_csv(NCU_CSV_PATH, encoding='utf-16', skiprows=skiprows, chunksize=1000)
        for chunk in chunk_iter:
            # Filter each chunk to only include metrics we care about
            if 'Metric Name' in chunk.columns:
                all_metrics = REQUIRED_METRIC_NAMES + OPTIONAL_METRIC_NAMES
                filtered_chunk = chunk[chunk['Metric Name'].isin(all_metrics)]
                if not filtered_chunk.empty:
                    chunk_list.append(filtered_chunk)
        
        # Combine all chunks
        if chunk_list:
            df_ncu_raw = pd.concat(chunk_list, ignore_index=True)
        else:
            print("No relevant metrics found in the CSV file")
            exit(1)
            
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        chunk_iter = pd.read_csv(NCU_CSV_PATH, encoding='latin-1', skiprows=skiprows, chunksize=1000)
        for chunk in chunk_iter:
            if 'Metric Name' in chunk.columns:
                all_metrics = REQUIRED_METRIC_NAMES + OPTIONAL_METRIC_NAMES
                filtered_chunk = chunk[chunk['Metric Name'].isin(all_metrics)]
                if not filtered_chunk.empty:
                    chunk_list.append(filtered_chunk)
        
        if chunk_list:
            df_ncu_raw = pd.concat(chunk_list, ignore_index=True)
        else:
            print("No relevant metrics found in the CSV file")
            exit(1)

    print(f"Successfully loaded '{NCU_CSV_PATH}' with {len(df_ncu_raw)} rows (filtered for relevant metrics).")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Nsight Compute report '{NCU_CSV_PATH}' not found.")
    print("Please run 'ncu --page details -o ncu_report --csv --force-overwrite ./harness.exe' first.")
    exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to read '{NCU_CSV_PATH}'. Error: {e}")
    exit(1)


def clean_kernel_name(name):
    """Cleans up the mangled kernel names from NCU."""
    # This regex is complex because template arguments can be nested
    # It tries to find the core name before the first '<'
    match = re.search(r'^(?:[a-zA-Z0-9_]+::)*([a-zA-Z0-9_]+)(?:<.*>)?$', str(name))
    if match:
        return match.group(1)
    return str(name)

# Ensure 'Metric Name' column exists
if 'Metric Name' not in df_ncu_raw.columns:
    print(f"CRITICAL ERROR: 'Metric Name' column not found in '{NCU_CSV_PATH}'.")
    print("Please ensure you exported the 'details' page from Nsight Compute.")
    print(f"Found columns: {df_ncu_raw.columns.tolist()}")
    exit(1)

# --- Pivoting the Data ---
try:
    # Filter for only the metrics we care about
    all_metrics = REQUIRED_METRIC_NAMES + OPTIONAL_METRIC_NAMES
    df_filtered = df_ncu_raw[df_ncu_raw['Metric Name'].isin(all_metrics)]

    # Convert Metric Value to numeric, coercing errors (like '%', ',') to NaN
    df_filtered['Metric Value'] = pd.to_numeric(df_filtered['Metric Value'], errors='coerce')

    # Pivot the table
    df_wide = df_filtered.pivot_table(
        index=KERNEL_NAME_COL,
        columns='Metric Name',
        values='Metric Value'
    ).reset_index()

    # Clean up kernel names
    # Create a simple name (e.g., 'saxpy') and a full name (e.g., '01_saxpy.cu')
    df_wide['kernel_name_simple'] = df_wide[KERNEL_NAME_COL].apply(clean_kernel_name)
    
    # Map simple name back to the original filename (e.g., 'saxpy' -> '01_saxpy.cu')
    # This relies on the kernel files being named correctly
    kernel_files = [f for f in os.listdir('kernels') if f.endswith('.cu')]
    name_map = {clean_kernel_name(f.replace('.cu', '')): f for f in kernel_files}
    
    df_wide['kernel_name'] = df_wide['kernel_name_simple'].map(name_map)
    
    # Handle any kernels that couldn't be mapped (e.g., if 'kernels/' dir is missing)
    df_wide['kernel_name'] = df_wide['kernel_name'].fillna(df_wide['kernel_name_simple'])

    # Set kernel_name as the index for easy lookup
    df_wide = df_wide.set_index('kernel_name')
    
    print("\nPivoted NCU Data:")
    print(df_wide[REQUIRED_METRIC_NAMES].head())

except KeyError:
    print("\nCRITICAL ERROR: Pivoting failed. This can happen if the required metric names are wrong.")
    print(f"We looked for: {REQUIRED_METRIC_NAMES}")
    print("Available unique metrics in the CSV:")
    print(df_ncu_raw['Metric Name'].unique()[:20]) # Show first 20 unique metrics
    exit(1)
except Exception as e:
    print(f"An error occurred during data pivoting: {e}")
    exit(1)


def classify_bottleneck(row):
    """Applies heuristics to classify a kernel based on its metrics."""
    # Get the required metrics, default to 0 if they are missing (e.g., NaN)
    sm_throughput = row.get(REQUIRED_METRIC_NAMES[0], 0)
    dram_throughput = row.get(REQUIRED_METRIC_NAMES[1], 0)

    # --- Classification Heuristics ---
    if sm_throughput > 60 and dram_throughput < 30:
        return "Compute-Bound"
    elif dram_throughput > 60 and sm_throughput < 30:
        return "Memory-Bound"
    elif sm_throughput < 20 and dram_throughput < 20:
        # Both are low, indicating a stall
        return "Latency-Bound"
    elif sm_throughput > 40 and dram_throughput > 40:
        # Both are significantly utilized
        return "MIXED(Compute/Memory)"
    elif sm_throughput > 40:
        return "Likely COMPUTE-BOUND"
    elif dram_throughput > 40:
        return "Likely MEMORY-BOUND"
    else:
        # Default fallback if no other rule matches
        return "MIXED/OTHER"

# Apply classification
df_wide['ground_truth_label'] = df_wide.apply(classify_bottleneck, axis=1)

# Save the derived ground truth
df_ground_truth = df_wide[['ground_truth_label'] + REQUIRED_METRIC_NAMES + [col for col in OPTIONAL_METRIC_NAMES if col in df_wide.columns]]
df_ground_truth.to_csv(OUTPUT_CSV_PATH)

print(f"\nSuccessfully processed NCU data and saved ground truth to '{OUTPUT_CSV_PATH}'")
print("\n--- Ground Truth Classification ---")
print(df_ground_truth[['ground_truth_label', REQUIRED_METRIC_NAMES[0], REQUIRED_METRIC_NAMES[1]]])


# --- Phase 2: Load LLM Predictions ---

print("\n--- Phase 2: Loading LLM Predictions ---")

try:
    df_llm = pd.read_csv(LLM_CSV_PATH)
    
    # Handle different column name possibilities
    if 'kernel' in df_llm.columns:
        df_llm = df_llm.rename(columns={'kernel': 'kernel_name'})
    
    # Clean kernel names to match ground truth (remove prefixes like "01_", "02_", etc.)
    def clean_llm_kernel_name(name):
        # Remove number prefixes like "01_", "02_", etc.
        cleaned = re.sub(r'^\d+_', '', str(name))
        # Remove .cu extension if present
        cleaned = cleaned.replace('.cu', '')
        return cleaned
    
    df_llm['kernel_name'] = df_llm['kernel_name'].apply(clean_llm_kernel_name)
    
    # Standardize LLM labels (e.g., "MEMORY-BOUND" -> "Memory-Bound")
    df_llm['label'] = df_llm['label'].str.title().str.replace("Compute-Bound", "Compute-Bound").str.replace("Memory-Bound", "Memory-Bound").str.replace("Latency-Bound", "Latency-Bound")
    df_llm = df_llm.set_index('kernel_name')
    print(f"Successfully loaded LLM predictions from '{LLM_CSV_PATH}'")
    print("LLM predictions with cleaned kernel names:")
    print(df_llm.head())

except FileNotFoundError:
    print(f"CRITICAL ERROR: LLM predictions file '{LLM_CSV_PATH}' not found.")
    print("Please run 'python scripts/predict_llms.py' first.")
    exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to read '{LLM_CSV_PATH}'. Error: {e}")
    exit(1)


# --- Phase 3: Merge & Compare ---

print("\n--- Phase 3: Merging Ground Truth and LLM Predictions ---")

# Merge the two dataframes on the kernel_name index
merged_df = df_ground_truth.join(df_llm, how='inner')

# Handle cases where LLM prediction might be missing or failed
merged_df['label'] = merged_df['label'].fillna('ERROR_PREDICTION')
merged_df = merged_df.rename(columns={'label': 'llm_label', 'rationale': 'llm_rationale'})

# Calculate accuracy
merged_df['is_correct'] = (merged_df['ground_truth_label'] == merged_df['llm_label'])
accuracy = merged_df['is_correct'].mean() * 100

print(f"\n--- Overall Accuracy: {accuracy:.2f}% ({merged_df['is_correct'].sum()} / {len(merged_df)}) ---")

# --- Full Classification Report ---
print("\n--- Classification Report ---")
# Get all unique labels from both ground truth and predictions
labels = np.unique(merged_df[['ground_truth_label', 'llm_label']].values)
report = classification_report(merged_df['ground_truth_label'], merged_df['llm_label'], labels=labels, zero_division=0)
print(report)

# --- Confusion Matrix ---
print("\n--- Generating Confusion Matrix Plot ---")
cm = confusion_matrix(merged_df['ground_truth_label'], merged_df['llm_label'], labels=labels)

# Set up the plot with simpler styling to avoid memory issues
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 8))

# Create the heatmap with simpler styling
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels.tolist(), yticklabels=labels.tolist(), 
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            square=True, ax=ax)

# Enhanced title and labels
ax.set_title('Confusion Matrix: LLM vs Ground Truth\nGPU Performance Bottleneck Classification', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label (LLM)', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Label (Ground Truth)', fontsize=12, fontweight='bold')

# Rotate labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Successfully saved enhanced 'confusion_matrix.png'")
plt.close()


# --- Mismatch Analysis ---
print("\n--- Mismatch Analysis ---")
mismatched_df = merged_df[~merged_df['is_correct']]
mismatched_df = mismatched_df[['ground_truth_label', 'llm_label', 'llm_rationale'] + REQUIRED_METRIC_NAMES]

if mismatched_df.empty:
    print("No mismatches found! Perfect classification.")
else:
    print(f"Found {len(mismatched_df)} mismatches. Saving details to '{MISMATCH_CSV_PATH}'")
    mismatched_df.to_csv(MISMATCH_CSV_PATH)
    
    # Print details of mismatches
    for kernel_name, row in mismatched_df.iterrows():
        print(f"\n[MISMATCH] Kernel: {kernel_name}")
        print(f"  > Ground Truth: {row['ground_truth_label']}")
        print(f"  > LLM Predicted:  {row['llm_label']}")
        print(f"  > LLM Rationale:  {row['llm_rationale']}")
        print(f"  > Hardware Data:  SM={row[REQUIRED_METRIC_NAMES[0]]:.1f}%, DRAM={row[REQUIRED_METRIC_NAMES[1]]:.1f}%")


# --- Phase 4: Plotting Performance Profile ---

# This is the new, improved function
def plot_comparison(df):
    """
    Plots a grouped bar chart comparing Compute (SM) Throughput and DRAM Throughput
    for each kernel, saving it to 'performance_profile_comparison.png'.
    """
    print("\n--- Phase 4: Generating Performance Profile Comparison Plot ---")
    
    # Ensure the required metrics are present
    if 'Compute (SM) Throughput' not in df.columns or 'DRAM Throughput' not in df.columns:
        print("Required metrics for plotting are missing. Skipping plot generation.")
        return

    # Create a DataFrame for plotting
    plot_df = df[['Compute (SM) Throughput', 'DRAM Throughput']].copy()
    
    # Clean kernel names for better labels (e.g., '01_saxpy.cu' -> '01_saxpy')
    plot_df.index = plot_df.index.str.replace(r'\.cu$', '', regex=True)

    # --- Enhanced Plotting ---
    
    # Set up the figure with better styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Define a beautiful color palette
    colors = ['#FF6B6B', '#4ECDC4']  # Coral red and teal
    
    # Create the bar plot
    plot_df.plot(
        kind='bar', 
        width=0.75,
        color=colors,
        ax=ax,
        edgecolor='white',
        linewidth=1
    )
    
    # Enhance the title and labels
    ax.set_title('GPU Kernel Performance Profile\nCompute vs. Memory Throughput Analysis', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Throughput (% of Peak Performance)', fontsize=14, fontweight='bold')
    ax.set_xlabel('CUDA Kernels', fontsize=14, fontweight='bold')
    
    # Customize tick labels
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)
    
    # Enhance legend
    ax.legend(['Compute (SM) Throughput', 'Memory (DRAM) Throughput'], 
              loc='upper right', fontsize=12, frameon=True, 
              title='Metric Type', title_fontsize=13)
    
    # Add reference lines
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.6)
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, max(plot_df.max()) * 1.1)
    
    # Enhance background
    fig.patch.set_facecolor('white')
    
    # Use tight_layout to ensure labels fit
    plt.tight_layout()
    
    # Save the figure
    plot_filename = 'performance_profile_comparison.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Successfully saved enhanced plot to '{plot_filename}'")
    plt.close(fig)

# Call the plotting function
plot_comparison(df_wide)

print("\n--- Analysis Complete ---")