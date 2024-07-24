import os
import requests
import json
import pandas as pd
from datetime import datetime
from transformers import GPT2Tokenizer
import re

# Environment setup
count_token_mode = "tiktoken"  # default value
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def count_chatgpt_tokens(model, prompt):
    tokens = tokenizer.encode(prompt)
    return len(tokens)

# Function to call ChatGPT API with functions
def ask_chatgpt_w_functions(messages, functions, model, url, api_key, max_tokens=500, debug=False):
    if debug:
        print("Debugging mode")
    
    is_azure = re.search(r".openai.azure.com/openai/deployments", url) is not None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}' if not is_azure else '',
        'api-key': api_key if is_azure else ''
    }

    body = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": functions,
        "tool_choice": "auto" if is_azure else None,
        "model": model if not is_azure else None,
        "function_call": "auto" if not is_azure else None
    }

    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        response_json = response.json()
        choices = response_json.get("choices", [])
        return pd.DataFrame([{
            "choices": choices,
            "id": response_json.get("id"),
            "created": datetime.fromtimestamp(response_json.get("created")),
            "model": response_json.get("model")
        }])
    else:
        if debug:
            print(response.content)
        response_json = response.json()
        return pd.DataFrame([{
            "error": response_json.get("error", {}).get("message"),
            "updated": datetime.now(),
            "model": model
        }])

# Function to call ChatGPT API without functions
def ask_chatgpt_wo_functions(messages, model, url, api_key, max_tokens=500, debug=False):
    if debug:
        print("Debugging mode")

    is_azure = re.search(r".openai.azure.com/openai/deployments", url) is not None
    is_mmc = re.search(r".mgti.mmc.com/coreapi/openai/v1/deployments", url) is not None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}' if not is_azure else '',
        'api-key': api_key if is_azure else ''
    }

    body = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": messages,
        "model": model if not (is_azure or is_mmc) else None
    }

    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        response_json = response.json()
        choices = response_json.get("choices", [])
        return pd.DataFrame([{
            "choices": choices,
            "id": response_json.get("id"),
            "created": datetime.fromtimestamp(response_json.get("created")),
            "model": response_json.get("model")
        }])
    else:
        response_json = response.json()
        return pd.DataFrame([{
            "error": response_json.get("error", {}).get("message"),
            "updated": datetime.now(),
            "model": model
        }])

def get_chatgpt_content_from_response(response):
    if response.empty:
        return ""
    if 'message.content' in response.columns:
        return response['message.content'].iloc[0]
    if isinstance(response['choices'].iloc[0], dict):
        return response['choices'].iloc[0].get('message', {}).get('content', '')
    return response['choices'][0][0].get('message', {}).get('content', '')

def display_prompt(prompt):
    for item in prompt:
        role = item.get('role')
        content = item.get('content')
        print(f"{role}: \033[1m\033[34m{content}\033[0m")

# # Example usage
# if __name__ == "__main__":
#     messages = [{"role": "user", "content": "Hello, how are you?"}]
#     functions = []  # Add functions if needed
#     model = "text-davinci-003"
#     url = "YOUR_OPENAI_API_ENDPOINT"
#     api_key = "YOUR_OPENAI_API_KEY"
    
#     response = ask_chatgpt_wo_functions(messages, model, url, api_key)
#     content = get_chatgpt_content_from_response(response)
#     display_prompt(messages)
#     print("Response Content:", content)
