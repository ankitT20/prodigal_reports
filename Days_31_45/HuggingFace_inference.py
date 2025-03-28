#pip install huggingface_hub
#export HF_TOKEN="<>"
import os
from huggingface_hub import InferenceClient
import json

with open(".env") as f:  
    os.environ.update(line.strip().split("=", 1) for line in f if "=" in line)  
HF_TOKEN = os.getenv("HF_TOKEN")

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


response=call_llm(llm_client, "write me a crazy joke")
print (response)
