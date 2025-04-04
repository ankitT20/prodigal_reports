#pip install huggingface_hub
import os
from huggingface_hub import InferenceClient
import json
from dotenv import load_dotenv

dotenv_path = r'C:\Users\ANKIT\Documents\VScode\prodigal_reports\.env'
load_dotenv(dotenv_path)
# Access the API key
HF_TOKEN = os.getenv('HF_TOKEN')

# with open(".env") as f:  
#     os.environ.update(line.strip().split("=", 1) for line in f if "=" in line)  
# HF_TOKEN = os.getenv("HF_TOKEN")

# repo_id = "meta-llama/Llama-3.2-1B"
repo_id = "meta-llama/Llama-3.2-1B-Instruct"

llm_client = InferenceClient(
    model=repo_id,
    token=HF_TOKEN,
    timeout=120,
)

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


response=call_llm(llm_client, "which model are you using?")
print (response)


""" OUTPUT:
PS C:\Users\ANKIT\Documents\VScode> & C:/Users/ANKIT/AppData/Local/Programs/Python/Python311/python.exe c:/Users/ANKIT/Documents/VScode/prodigal_reports/Days_31_45/HuggingFace_inference.py
which model are you using? I can help you with any questions you may have.

## Step 1: Identify the type of question
The problem asks for information about a specific model, which implies that there is a particular model in question.

## Step 2: Determine the context of the question
Since the question is asking for information about a specific model, it is likely related to a particular application, industry, or field.

## Step 3: Consider the possible answers
Without specific information about the model, it is impossible to provide a precise answer. However, I can suggest some possible models that are commonly used in various fields.

## Step 4: Provide a general response
If you could provide more context or information about the model you are using, I would be happy to try and assist you further.

The final answer is: $\boxed{N/A}$
PS C:\Users\ANKIT\Documents\VScode> 
"""
