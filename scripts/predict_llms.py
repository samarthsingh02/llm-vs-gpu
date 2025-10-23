import google.generativeai as genai
import os
import json
import pandas as pd
import time
import re
from dotenv import load_dotenv
import google.api_core.exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv() # Loads variables from .env into environment

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
    "max_output_tokens": 1024,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
generation_config=generation_config,
    safety_settings={ # Add this block
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Prompt Template
PROMPT_TEMPLATE = """
You are an expert GPU performance engineer. Your task is to classify a given CUDA kernel based on its primary performance bottleneck.
The GPU has the following specifications: {gpu_flops} TFLOPs FP32 compute, {gpu_bw} GB/s memory bandwidth.

Classify the kernel as one of: COMPUTE-BOUND, MEMORY-BOUND, or LATENCY-BOUND.

Return your response ONLY in JSON format, with no other text before or after the JSON block. The JSON should have two keys: \"label\" and \"rationale\".

Example response:
{{
    "label": "MEMORY-BOUND",
    "rationale": "The kernel performs a simple vector addition, reading two floats and writing one for each thread. This results in a low operational intensity (FLOPs/Byte), meaning performance is limited by memory bandwidth, not compute."
}}

---
KERNEL CODE:
```cuda
{kernel_code}
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
        result = json.loads(json_str)
        # If result is a string, try to parse again (handles double-encoded JSON)
        if isinstance(result, str):
            try:
                result2 = json.loads(result)
                return result2
            except Exception:
                pass
        return result
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

    # --- Start Replacement Block ---
    try:
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        print("  -> Sending request to Generative AI API...")
        # Use .format() correctly, assuming PROMPT_TEMPLATE is now a regular string
        prompt = PROMPT_TEMPLATE.format(gpu_flops=GPU_FP32_TFLOPS, gpu_bw=GPU_MEM_BW_GBPS, kernel_code=kernel_code)

        response = model.generate_content(prompt) # Make the API call

# --- NEW: Process response *after* checking validity ---
        llm_output_text = "" # Initialize empty
        parse_error_reason = "Unknown"
        parsed_json = None
        api_call_successful = False # Flag to track if API returned normally

        # 1. Check for prompt blocking first
        if response.prompt_feedback.block_reason:
            block_reason_str = response.prompt_feedback.block_reason.name
            print(f"  -> ERROR: Prompt was blocked. Reason: {block_reason_str}")
            parse_error_reason = f"Prompt blocked: {block_reason_str}"

        # 2. Check if candidates exist
        elif not response.candidates:
             print("  -> ERROR: No candidates returned by the API.")
             parse_error_reason = "No candidates in response."

        # 3. Check candidate finish reason ONLY IF candidates exist
        elif response.candidates[0].finish_reason != 1: # 1 = FINISH_REASON_STOP (Normal)
            finish_reason_map = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
            reason_code = response.candidates[0].finish_reason
            reason_str = finish_reason_map.get(reason_code, f"UNKNOWN ({reason_code})")
            print(f"  -> ERROR: Content generation stopped abnormally. Finish Reason: {reason_str}")
            parse_error_reason = f"Generation stopped: {reason_str}"
            # Try to get partial text IF available (might be empty if blocked for safety)
            try:
                # Check if parts exist before accessing text
                if response.candidates[0].content.parts:
                    llm_output_text = response.text
                    print("  -> WARNING: Attempting to parse partial/stopped response text.")
                else:
                    print("  -> ERROR: Response stopped with no content parts.")
                    llm_output_text = ""
            except Exception as e: # Catch any other issues accessing partial text
                print(f"  -> ERROR: Cannot access response text (reason: {e}).")
                llm_output_text = "" # Ensure it's empty
        else:
            # 4. Normal case: Generation finished correctly, API call was successful
            api_call_successful = True
            print("  -> Response received normally. Attempting to parse JSON...")
            try:
                # Check parts exist before accessing text
                if response.candidates[0].content.parts:
                    llm_output_text = response.text
                else:
                     print("  -> ERROR: Successful finish but no content parts found.")
                     parse_error_reason = "Successful finish but no content."
                     llm_output_text = ""
            except Exception as e:
                 print(f"  -> ERROR: Accessing response.text failed unexpectedly: {e}")
                 parse_error_reason = f"Error accessing response text: {e}"
                 llm_output_text = "" # Ensure empty on error

        # 5. Attempt parsing only if we have some text
        if llm_output_text:
            parsed_json = parse_json_response(llm_output_text)
            print(f"    DEBUG: LLM output type: {type(llm_output_text)}")
            print(f"    DEBUG: Parsed JSON type: {type(parsed_json)}")
            print(f"    DEBUG: Parsed JSON value: {parsed_json}")
        elif api_call_successful: # If API call finished ok but parsing wasn't attempted
             print(f"    DEBUG: No valid text content received to parse.")
             parse_error_reason = "No text content received"


        # 6. Check final result and append
        if isinstance(parsed_json, dict) and "label" in parsed_json and "rationale" in parsed_json:
             results.append({
                 "kernel": kernel_name,
                 "label": parsed_json["label"],
                 "rationale": parsed_json["rationale"]
             })
             print(f"  -> SUCCESS: Parsed prediction: {parsed_json['label']}")
        else: # Handles API errors, blocks, parse failures, or missing keys
             print(f"  -> FAILED: Could not get valid prediction.")
             # Construct a more informative rationale
             fail_rationale_parts = []
             if parse_error_reason != "Unknown":
                 fail_rationale_parts.append(f"API/Response Issue: {parse_error_reason}")
             if parsed_json is None and llm_output_text: # Parsing failed on existing text
                 fail_rationale_parts.append(f"Raw text (may be incomplete/invalid JSON): {llm_output_text}")
             elif not llm_output_text and parse_error_reason == "Unknown": # Should not happen often
                 fail_rationale_parts.append("Unknown failure before parsing.")

             fail_rationale = ". ".join(fail_rationale_parts)
             if not fail_rationale: # Failsafe
                 fail_rationale = "Unknown error during processing."


             results.append({
                 "kernel": kernel_name,
                 # Prioritize API/block errors over parse errors for labeling
                 "label": "API_ERROR" if parse_error_reason != "Unknown" else "PARSE_ERROR",
                 "rationale": fail_rationale
             })

    # --- Keep the specific exception handling for Rate Limits etc. ---
    except google.api_core.exceptions.ResourceExhausted as e: # Catch 429 specifically
        print(f"  -> RATE LIMIT ERROR: {e}")
        retry_seconds = 60 # Default wait time
        match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(e), re.IGNORECASE)
        if match:
            try:
                retry_seconds = int(match.group(1)) + 5 # Add a 5 sec buffer
                print(f"  -> API suggests retrying in {match.group(1)}s. Waiting {retry_seconds}s...")
            except ValueError:
                print(f"  -> Could not parse retry delay. Waiting default {retry_seconds}s...")
        else:
            print(f"  -> No retry delay suggestion. Waiting default {retry_seconds}s...")

        results.append({
            "kernel": kernel_name,
            "label": "RATE_LIMIT_ERROR",
            "rationale": f"Hit API rate limit. Wait {retry_seconds}s applied. Error: {e}"
        })
        time.sleep(retry_seconds) # Wait before processing the NEXT kernel
        continue # Skip to the next kernel after waiting

    except google.api_core.exceptions.GoogleAPICallError as e: # Catch other API call errors
        print(f"  -> ERROR calling Generative AI API (gRPC/HTTP Error): {e}")
        results.append({"kernel": kernel_name, "label": "API_CALL_ERROR", "rationale": str(e)})
        time.sleep(1)
        continue

    except Exception as e: # Catch other unexpected errors
        print(f"  -> ERROR during API call or response processing: {e}")
        import traceback
        traceback.print_exc()
        results.append({"kernel": kernel_name, "label": "UNEXPECTED_SCRIPT_ERROR", "rationale": str(e)})
        time.sleep(1)
        continue
    # --- End Replacement Block ---
    
    time.sleep(1)  # Rate limit protection between successful requests

# Save results to CSV
df = pd.DataFrame(results)
try:
    df.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to {output_csv_file}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

print("LLM prediction process finished.")
