import pandas as pd
import os

# --- Configuration ---
NCU_CSV_PATH = 'ncu_report_details.csv'
OUTPUT_CSV_PATH = 'data/derived_ground_truth.csv'
LLM_CSV_PATH = 'data/llm_predictions.csv'

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
    with open(NCU_CSV_PATH, 'r', encoding='utf-16') as f:
        lines = f.readlines()
    skiprows = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('"ID"'):
             skiprows = i
             break
        if i > 20:
             print("Warning: Could not reliably determine header row start. Assuming 0.")
             skiprows = 0
             break

    print(f"Attempting to read CSV starting from row {skiprows} with UTF-16 encoding...")
    ncu_df_long = pd.read_csv(NCU_CSV_PATH, encoding='utf-16', skiprows=skiprows)

except (UnicodeDecodeError, FileNotFoundError) as e:
    print(f"Failed to read with UTF-16 (Error: {e}). Trying latin-1...")
    try:
        with open(NCU_CSV_PATH, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        skiprows = 0
        for i, line in enumerate(lines):
             if line.strip().startswith('"ID"'):
                 skiprows = i
                 break
             if i > 20:
                 print("Warning: Could not reliably determine header row start. Assuming 0.")
                 skiprows = 0
                 break
        print(f"Attempting to read CSV starting from row {skiprows} with latin-1 encoding...")
        ncu_df_long = pd.read_csv(NCU_CSV_PATH, encoding='latin-1', skiprows=skiprows)
    except Exception as e_inner:
        print(f"CRITICAL ERROR: Error reading CSV with fallback encoding: {e_inner}")
        exit()

# Filter for the metrics we need
all_needed_metrics = REQUIRED_METRIC_NAMES + OPTIONAL_METRIC_NAMES
ncu_df_filtered = ncu_df_long[ncu_df_long['Metric Name'].isin(all_needed_metrics)].copy()

# *** ADDED DEBUGGING PRINT ***
print("\nUnique metric names found after filtering (before pivot):")
if ncu_df_filtered.empty:
    print("  >> DataFrame is empty - No specified metrics were found!")
    print("  >> Check the 'Metric Name' column in ncu_report_details.csv for the correct names.")
    print(f"  >> Names currently searched for: {all_needed_metrics}")
else:
    print(ncu_df_filtered['Metric Name'].unique())
print("-" * 30)
# *** END DEBUGGING PRINT ***

# Exit if no required metrics were found at all
if ncu_df_filtered[ncu_df_filtered['Metric Name'].isin(REQUIRED_METRIC_NAMES)].empty:
     print("CRITICAL ERROR: None of the required metric names were found in the CSV.")
     print(f"Required names searched for: {REQUIRED_METRIC_NAMES}")
     # As a fallback, try to detect close alternatives that sometimes appear
     # and guide the user.
     print("Here are some sample metric names detected in your file:")
     try:
         print(ncu_df_long['Metric Name'].dropna().unique()[:25])
     except Exception:
         pass
     exit()


# Convert Metric Value to numeric, handling commas and errors
ncu_df_filtered['Metric Value'] = ncu_df_filtered['Metric Value'].astype(str).str.replace(',', '', regex=False)
ncu_df_filtered['Metric Value'] = pd.to_numeric(ncu_df_filtered['Metric Value'], errors='coerce')

# --- Pivot the table ---
print("Pivoting the table to get metrics as columns...")
try:
    ncu_df_wide = ncu_df_filtered.pivot_table(
        index=KERNEL_NAME_COL,
        columns='Metric Name',
        values='Metric Value',
        aggfunc='first'
    )
except KeyError as e:
     print(f"CRITICAL ERROR during pivot: Missing column - {e}. Check KERNEL_NAME_COL.")
     print("Columns available:", list(ncu_df_filtered.columns))
     exit()

# Check if required metric columns exist after pivot
missing_required = [col for col in REQUIRED_METRIC_NAMES if col not in ncu_df_wide.columns]
if missing_required:
    print("\nCRITICAL ERROR: Required metric columns missing after pivoting:")
    for col in missing_required:
        print(f"- {col}")
    print("Available columns after pivot:", list(ncu_df_wide.columns))
    print("This might happen if the metrics existed but had non-numeric values that became NaN.")
    exit()

# Report on optional metrics
found_optional = [col for col in OPTIONAL_METRIC_NAMES if col in ncu_df_wide.columns]
missing_optional = [col for col in OPTIONAL_METRIC_NAMES if col not in ncu_df_wide.columns]
if found_optional:
    print("\nFound optional latency metrics:")
    for col in found_optional: print(f"- {col}")
if missing_optional:
    print("\nDid not find optional latency metrics:")
    for col in missing_optional: print(f"- {col}")

# --- Define Bottleneck Classification Function ---
def classify_bottleneck(row):
    sm_usage = row.get(REQUIRED_METRIC_NAMES[0], 0)
    dram_usage = row.get(REQUIRED_METRIC_NAMES[1], 0)

    if pd.isna(sm_usage) or pd.isna(dram_usage):
        return "UNKNOWN_DATA_ISSUE"

    if sm_usage > 60 and dram_usage < 30:
        return "COMPUTE-BOUND"
    elif dram_usage > 60 and sm_usage < 30:
        return "MEMORY-BOUND"
    elif sm_usage < 40 and dram_usage < 40:
        return "LATENCY-BOUND"
    elif sm_usage > 50 and dram_usage > 50:
        return "MIXED(Compute/Memory)"
    elif sm_usage >= 40 and dram_usage >= 40:
         return "MIXED/OTHER"
    elif sm_usage > dram_usage:
         return "Likely COMPUTE-BOUND"
    else:
         return "Likely MEMORY-BOUND"


# Apply classification
ncu_df_wide['ground_truth_label'] = ncu_df_wide.apply(classify_bottleneck, axis=1)

# Prepare final ground truth DataFrame
ground_truth_data = ncu_df_wide.reset_index()
ground_truth_data.rename(columns={KERNEL_NAME_COL: 'kernel_name'}, inplace=True)

print("\nGround Truth DataFrame Head (after pivoting):")
display_cols = ['kernel_name', 'ground_truth_label'] + [col for col in REQUIRED_METRIC_NAMES if col in ground_truth_data.columns] # Use list comprehension for safety
print(ground_truth_data[display_cols].head())

# Save derived ground truth
try:
    if not os.path.exists('data'):
        os.makedirs('data')
    ground_truth_data.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nDerived ground truth saved to {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"\nError saving derived ground truth: {e}")

print("\n--- Phase 1 Complete ---")

# --- Phase 2: LLM Prediction Processing ---
# (Code remains the same as previous version)
print("\n--- Phase 2: Processing LLM Predictions ---")
try:
    df_llm = pd.read_csv(LLM_CSV_PATH)

    if df_llm['kernel'].str.contains(r'^\d+_').any():
         print("Standardizing LLM kernel names (removing prefix like '01_')...")
         df_llm['kernel_name'] = df_llm['kernel'].str.split('_', n=1).str[1]
    else:
         print("LLM kernel names seem standard, using as is.")
         df_llm['kernel_name'] = df_llm['kernel']

    error_labels = ['API_ERROR', 'ERROR', 'FILE_NOT_FOUND', 'UNEXPECTED_ERROR']
    df_llm['llm_label'] = df_llm['label'].apply(lambda x: 'ERROR_PREDICTION' if isinstance(x, str) and x.upper() in error_labels else x)
    df_llm['llm_label'] = df_llm['llm_label'].astype(str).str.upper()

    print("\nLLM Predictions DataFrame Head (cleaned):")
    print(df_llm[['kernel_name', 'llm_label', 'rationale']].head())
    print("\n--- Phase 2 Complete ---")

except FileNotFoundError:
    print(f"CRITICAL ERROR: LLM predictions file not found at {LLM_CSV_PATH}")
    exit()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to process LLM predictions: {e}")
    exit()


# --- Phase 3: Comparative Analysis ---
# (Code remains the same as previous version)
print("\n--- Phase 3: Comparing Ground Truth and LLM Predictions ---")

df_llm_subset = df_llm[['kernel_name', 'llm_label', 'rationale']]
gt_display_cols = ['kernel_name', 'ground_truth_label'] + [col for col in REQUIRED_METRIC_NAMES + found_optional if col in ground_truth_data.columns]
df_gt_subset = ground_truth_data[gt_display_cols]

merged_df = pd.merge(df_gt_subset, df_llm_subset, on='kernel_name', how='inner')

if merged_df.empty:
    print("\nCRITICAL ERROR: Merging failed. No common kernel names found.")
    print("Ground truth kernel names:", df_gt_subset['kernel_name'].unique())
    print("LLM kernel names:", df_llm_subset['kernel_name'].unique())
    exit()
else:
     print(f"\nSuccessfully merged {len(merged_df)} kernels.")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

all_labels_gt = sorted(merged_df['ground_truth_label'].unique())
all_labels_llm = sorted(merged_df['llm_label'].unique())
all_labels = sorted(list(set(all_labels_gt) | set(all_labels_llm)))
if "ERROR_PREDICTION" in all_labels:
     all_labels.remove("ERROR_PREDICTION")
     all_labels.append("ERROR_PREDICTION")

y_true = merged_df['ground_truth_label']
y_pred = merged_df['llm_label']

accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as confusion_matrix.png")
# plt.show()

mismatched = merged_df[merged_df['ground_truth_label'] != merged_df['llm_label']]

print("\n--- Mismatch Analysis ---")
if mismatched.empty:
    print("No mismatches found!")
else:
    mismatch_details = []
    for index, row in mismatched.iterrows():
        # Safely get metric values, providing 'N/A' if the column doesn't exist
        sm_metric_val = row.get(REQUIRED_METRIC_NAMES[0], float('nan'))
        dram_metric_val = row.get(REQUIRED_METRIC_NAMES[1], float('nan'))

        detail = {
            'Kernel': row['kernel_name'],
            'Ground Truth': row['ground_truth_label'],
            'LLM Predicted': row['llm_label'],
            'LLM Rationale': row['rationale'],
            f"Nsight SM ({REQUIRED_METRIC_NAMES[0]})": f"{sm_metric_val:.1f}%" if not pd.isna(sm_metric_val) else 'N/A',
            f"Nsight DRAM ({REQUIRED_METRIC_NAMES[1]})": f"{dram_metric_val:.1f}%" if not pd.isna(dram_metric_val) else 'N/A'
        }
        
        # Add optional metrics if they exist
        for col in found_optional:
            # Use the full metric name found in the CSV as the key
            metric_val_opt = row.get(col, float('nan')) # Get the value safely
            # Format the value for printing/saving
            formatted_val = f"{metric_val_opt:.1f}" if not pd.isna(metric_val_opt) else 'N/A'
            # Check if '%' should be added based on Metric Unit (Requires loading Metric Unit earlier)
            # For simplicity now, we'll just add the value. You might refine this later.
            detail[f"Nsight Opt ({col})"] = formatted_val # Use full name
            
        mismatch_details.append(detail)
        print(f"\nKernel: {detail['Kernel']}")
        print(f"  Ground Truth: {detail['Ground Truth']}")
        print(f"  LLM Predicted: {detail['LLM Predicted']}")
        print(f"  LLM Rationale: {str(detail['LLM Rationale'])[:200]}...")
        print(f"  Nsight SM: {detail[f'Nsight SM ({REQUIRED_METRIC_NAMES[0]})']}")
        print(f"  Nsight DRAM: {detail[f'Nsight DRAM ({REQUIRED_METRIC_NAMES[1]})']}")
        # Print optional metrics found
        for col in found_optional:
             if f"Nsight Opt ({col})" in detail: # Check if key exists
                 print(f"  Nsight Opt ({col}): {detail[f'Nsight Opt ({col})']}")
        print("-" * 20)

    try:
        mismatch_df = pd.DataFrame(mismatch_details)
        mismatch_df.to_csv('data/mismatched_predictions.csv', index=False)
        print("\nMismatch details saved to data/mismatched_predictions.csv")
    except Exception as e:
        print(f"\nError saving mismatch details: {e}")

print("\nGenerating Roofline-style plot...")
plt.figure(figsize=(12, 8))
color_map = {
    'COMPUTE-BOUND': 'red', 'MEMORY-BOUND': 'blue', 'LATENCY-BOUND': 'green',
    'MIXED(Compute/Memory)': 'purple', 'MIXED/OTHER': 'orange',
    'Likely COMPUTE-BOUND': 'lightcoral', 'Likely MEMORY-BOUND': 'lightblue',
    'ERROR_PREDICTION': 'black', 'UNKNOWN_DATA_ISSUE': 'grey'
    }

# Ensure required columns exist and drop rows with NaN in them before plotting
plot_df = merged_df.dropna(subset=[REQUIRED_METRIC_NAMES[0], REQUIRED_METRIC_NAMES[1]])

sns.scatterplot(
    data=plot_df, x=REQUIRED_METRIC_NAMES[0], y=REQUIRED_METRIC_NAMES[1],
    hue='ground_truth_label', palette=color_map, style='llm_label',
    s=150, alpha=0.8
)

for i in range(plot_df.shape[0]):
     plt.text(plot_df[REQUIRED_METRIC_NAMES[0]].iloc[i] + 0.7,
              plot_df[REQUIRED_METRIC_NAMES[1]].iloc[i],
              plot_df['kernel_name'].iloc[i], fontsize=9)

plt.title('Kernel Performance Profile (Ground Truth Hue, LLM Prediction Marker Style)')
plt.xlabel(f'SM Throughput ({REQUIRED_METRIC_NAMES[0]})') # Use actual name
plt.ylabel(f'DRAM Throughput ({REQUIRED_METRIC_NAMES[1]})') # Use actual name
plt.xlim(-5, 105); plt.ylim(-5, 105); plt.grid(True)
plt.legend(title='Bottleneck (GT Hue / LLM Style)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=(0, 0, 0.85, 1))
plt.savefig('performance_profile_comparison.png')
print("Performance profile plot saved as performance_profile_comparison.png")
# plt.show()

print("\n--- Phase 3 Complete ---")
print("\nAnalysis finished. Check generated CSV files and PNG plots.")