import requests

# Define the API endpoint
api_url = "https://api.aimlapi.com/v1/chat/completions"

import os
from dotenv import load_dotenv

# Set your AI/ML API key
dotenv_path = r'C:\Users\ANKIT\Documents\VScode\prodigal_reports\.env'
load_dotenv(dotenv_path)
# Access the API key
API_KEY = os.getenv('API_KEY')

# Configure the request headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create the payload with the desired model and prompt
payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "which model are you using? GPT-4o or GPT-4o-mini? or something else?"},
    ]
}

# Send the POST request to the API
response = requests.post(api_url, headers=headers, json=payload)

# Parse the response JSON
response_data = response.json()

# Extract and print the assistant's reply
assistant_reply = response_data["choices"][0]["message"]["content"]
print("Assistant:", assistant_reply)

"""OUTPUT:
PS C:\Users\ANKIT\Documents\VScode> & C:/Users/ANKIT/AppData/Local/Programs/Python/Python311/python.exe c:/Users/ANKIT/Documents/VScode/prodigal_reports/Days_31_45/aimlapi.py
Assistant: I am based on OpenAI's GPT-4 model. However, the specific variant, whether it might be designated as "GPT-4o," "GPT-4o-mini," or any other specific internal tag, is not disclosed in the available information. My capabilities are broadly aligned with what is generally known about GPT-4, which is designed to assist with a wide range of questions and tasks. If you have any questions or need help with something specific, feel free to ask!
PS C:\Users\ANKIT\Documents\VScode> 
"""
