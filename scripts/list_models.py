import google.generativeai as genai
import os

# --- Make sure your API key is set as an environment variable ---
# In PowerShell: $env:GOOGLE_API_KEY="YOUR_API_KEY"
# In Git Bash/WSL: export GOOGLE_API_KEY="YOUR_API_KEY"
# Or replace os.getenv("GOOGLE_API_KEY") below with your actual key string
# ---

try:
    api_key = "GOOGLE_API_KEY"
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring API key: {e}")
    exit()

print("Available models supporting 'generateContent':")
print("-" * 40)
try:
    for m in genai.list_models():
      # Check if the model supports the 'generateContent' method
      if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")
        # print(f"  Display Name: {m.display_name}")
        # print(f"  Description: {m.description[:60]}...") # Print start of description
        # print("-" * 20)
except Exception as e:
    print(f"An error occurred while listing models: {e}")

print("-" * 40)
print("Please choose a suitable model name (like 'gemini-pro' or 'gemini-1.0-pro') from the list above.")