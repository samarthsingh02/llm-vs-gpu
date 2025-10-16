import google.generativeai as genai
import os
import json
import pandas as pd

# Configure the API key
try:
    # IMPORTANT: REPLACE "YOUR_API_KEY" with your actual Google AI Studio API key.
    genai.configure(api_key="YOUR_API_KEY")
except Exception as e:
    print(f"Error configuring API key: {e}")
    print("Please replace 'YOUR_API_KEY' with your actual API key.")
    exit()

# Model configuration
generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config)

# Prompt Template
PROMPT_TEMPLATE = """
You are an expert GPU performance engineer. Your task is to classify a given CUDA kernel based on its primary performance bottleneck.
The GPU has the following specifications: 29 TFLOPs FP32 compute, 760 GB/s memory bandwidth.

Classify the kernel as one of: COMPUTE-BOUND, MEMORY-BOUND, or LATENCY-BOUND.

Return your response ONLY in JSON format, with no other text before or after the JSON block. The JSON should have two keys: "label" and "rationale".

Example response:
{
  "label": "MEMORY-BOUND",
  "rationale": "The kernel performs a simple vector addition, reading two floats and writing one for each thread. This results in a low operational intensity (FLOPs/Byte), meaning performance is limited by memory bandwidth, not compute."
}

---
KERNEL CODE:
```cuda
{kernel_code}