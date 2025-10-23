import google.generativeai as genai
import os
import json
import pandas as pd
import time
import re

# Configure the API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key == "YOUR_API_KEY":
        print("Warning: Using placeholder API key. Replace 'YOUR_API_KEY'.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring API key: {e}")
    print("Please ensure your API key is correct and valid.")
    exit()

# --- UPDATE GPU Specs Here ---
GPU_FP32_TFLOPS = 9.0  # From your hw_specs.json
GPU_MEM_BW_GBPS = 192  # From your hw_specs.json
# --- END UPDATE ---

# Model configuration
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config
)

# Prompt Template
PROMPT_TEMPLATE = f"""
You are an expert GPU performance engineer. Your task is to classify a given CUDA kernel based on its primary performance bottleneck.
The GPU has the following specifications: {GPU_FP32_TFLOPS} TFLOPs FP32 compute, {GPU_MEM_BW_GBPS} GB/s memory bandwidth.

Classify the kernel as one of: COMPUTE-BOUND, MEMORY-BOUND, or LATENCY-BOUND.

Return your response ONLY in JSON format, with no other text before or after the JSON block. The JSON should have two keys: "label" and "rationale".

Example response:
{{
  "label": "MEMORY-BOUND",
  "rationale": "The kernel performs a simple vector addition, reading two floats and writing one for each thread. This results in a low operational intensity (FLOPs/Byte), meaning performance is limited by memory bandwidth, not compute."
}}

---
KERNEL CODE:
```cuda
{{kernel_code}}
```
"""

# --- UPDATE Kernel File Paths Here ---
kernel_files = [
    "kernels/01_saxpy.cu",
    "kernels/02_naive_matmul.cu",
    "kernels/03_naive_transpose.cu",
    "kernels/04_tiled_transpose.cu",
    "kernels/05_bank_conflict.cu",
    "kernels/06_strided_global.cu",
    "kernels/07_branch_divergence.cu",
    "kernels/08_atomic_histogram.cu",
    "kernels/09_high_reg_pressure.cu",
    "kernels/10_parallel_reduction.cu",
    "kernels/11_tiled_matmul.cu"
]
# --- END UPDATE ---

# --- UPDATE Output File Path Here ---
output_csv_file = "data/llm_predictions.csv"
# --- END UPDATE ---

# Function to safely parse JSON from LLM response
def parse_json_response(text):
    json_str = None
    parsed_json = None

    # 1. Try to find JSON block using regex (handles ```json ... ```)
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip() # Strip the captured group
        print("    DEBUG: Found JSON block via regex.")
        try:
            parsed_json = json.loads(json_str)
            return parsed_json # Return early if regex worked
        except json.JSONDecodeError as e:
            print(f"    DEBUG: Error decoding JSON from regex match: {e}")
            json_str = None # Reset if regex block failed parsing

    # 2. Fallback: Find the substring from the first '{' to the last '}'
    if json_str is None: # Only proceed if regex failed or parsing regex failed
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1].strip() # Extract and strip immediately
            print("    DEBUG: Found JSON block via find('{') and rfind('}').")
        else:
            # 3. Last Resort: If no braces found, use the whole text (after stripping)
             print("    DEBUG: Could not find '{' and '}'. Treating stripped whole text as JSON.")
             json_str = text.strip()

    # Ensure json_str is never None (shouldn't happen with the logic above, but safety first)
    if json_str is None:
         print("    DEBUG: json_str is None after extraction attempts!")
         json_str = text.strip() # Failsafe

    try:
        # Final attempt to parse whatever json_str contains
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"    Error decoding JSON: {e}")
        print(f"    Attempted to parse final:\n---\n{json_str}\n---")
        print(f"    Original raw response was:\n---\n{text}\n---")
        return None
                
# --- Main Logic ---
results = []

output_dir = os.path.dirname(output_csv_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Starting LLM predictions for {len(kernel_files)} kernels...")

for kernel_path in kernel_files:
    kernel_name = os.path.basename(kernel_path).replace('.cu', '')
    print(f"Processing kernel: {kernel_name} ({kernel_path})")

    try:
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        prompt = PROMPT_TEMPLATE.format(kernel_code=kernel_code)

        try:
            response = model.generate_content(prompt)
            llm_output_text = response.text

            parsed_json = parse_json_response(llm_output_text)

            if parsed_json and "label" in parsed_json and "rationale" in parsed_json:
                results.append({
                    "kernel": kernel_name,
                    "label": parsed_json["label"],
                    "rationale": parsed_json["rationale"]
                })
                print(f"  -> Prediction: {parsed_json['label']}")
            else:
                print(f"  -> Failed to parse valid JSON response.")
                results.append({
                    "kernel": kernel_name,
                    "label": "ERROR",
                    "rationale": f"Failed to parse JSON. Raw response: {llm_output_text}"
                })

        except Exception as e:
            print(f"  -> Error calling Generative AI API: {e}")
            results.append({
                "kernel": kernel_name,
                "label": "API_ERROR",
                "rationale": str(e)
            })

        time.sleep(1)

    except FileNotFoundError:
        print(f"  -> Error: Kernel file not found at {kernel_path}")
        results.append({
            "kernel": kernel_name,
            "label": "FILE_NOT_FOUND",
            "rationale": f"File not found at {kernel_path}"
        })
    except Exception as e:
        print(f"  -> An unexpected error occurred: {e}")
        results.append({
            "kernel": kernel_name,
            "label": "UNEXPECTED_ERROR",
            "rationale": str(e)
        })

# Save results to CSV
df = pd.DataFrame(results)
try:
    df.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to {output_csv_file}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

print("LLM prediction process finished.")
