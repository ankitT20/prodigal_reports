import requests
import json

# Define the API endpoint
api_url = "http://localhost:11434/api/chat"

# Specify the model name
model_name = "llama3.2:1b"

# Construct the messages for the conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! which model are you using?"}
]

# Create the payload
payload = {
    "model": model_name,
    "messages": messages,
    "stream": False  # Set to True for streaming responses
}

# Send the POST request to the Ollama chat endpoint
response = requests.post(api_url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    response_data = response.json()
    assistant_reply = response_data.get("message", {}).get("content", "")
    print("Assistant:", assistant_reply)
else:
    print(f"Error: {response.status_code} - {response.text}")


"""OUTPUT:
PS C:\Users\ANKIT\Documents\VScode> & C:/Users/ANKIT/AppData/Local/Programs/Python/Python311/python.exe c:/Users/ANKIT/Documents/VScode/prodigal_reports/Days_31_45/Ollama.py
Assistant: I'm not using a specific model, as I'm a text-based AI assistant trained on a large language model. My responses are generated based on the patterns and associations in the data I was trained on, rather than relying on a particular software or hardware architecture. This allows me to respond quickly and easily, without the need for complex computations or specialized models."""

# The code sends a POST request to the Ollama API to get a response from the specified model (Llama 3.2:1b) based on the user's input.  
# Ollama is running inside Docker, and the API is exposed to host(windows) via localhost on port 11434 port mapping.
# To minimize RAM usage and avoid system crashes, Docker Engine is running inside Windows Subsystem for Linux (WSL) 2 Ubuntu 24.04.2 LTS.
""" PS C:\Users\ANKIT> wsl
ubuntu@MYWORLD:~$ docker exec -it ollama ollama list
NAME           ID              SIZE      MODIFIED
llama3.2:1b    baf6a787fdff    1.3 GB    4 hours ago"""

""" 
ubuntu@MYWORLD:~$ docker ps
CONTAINER ID   IMAGE                                COMMAND               CREATED       STATUS                        PORTS                                             NAMES
cedb087abc44   ghcr.io/open-webui/open-webui:main   "bash start.sh"       4 hours ago   Up About a minute (healthy)                                                     open-webui
2051ffaf025d   ollama/ollama                        "/bin/ollama serve"   4 hours ago   Up About a minute             0.0.0.0:11434->11434/tcp, [::]:11434->11434/tcp   ollama


ubuntu@MYWORLD:~$ docker stats
CONTAINER ID   NAME         CPU %     MEM USAGE / LIMIT    MEM %     NET I/O          BLOCK I/O   PIDS
cedb087abc44   open-webui   7.90%     630.5MiB / 7.76GiB   7.93%     0B / 0B          0B / 0B     20
2051ffaf025d   ollama       285.74%   1.805GiB / 7.76GiB   23.26%    75.6kB / 167kB   0B / 0B     22
"""
