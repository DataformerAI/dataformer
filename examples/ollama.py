from dataformer.llms.openllm import OpenLLM

# Enter Ollama local or runpod URL
url = "https://j0sxxemj0k3cbf-11434.proxy.runpod.net/api/chat"
model = "tinyllama"

llm = OpenLLM(None, url, "ollama", model)  # API key is not needed for ollama

prompt = "What is the difference between a frog and a toad?"

# Add stream = False in request
request_list = [{"messages": [{"role": "user", "content": prompt}], "stream": False}]


response_list = llm.generate(request_list)

for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["message"]["content"]
    tokens = response[1]["prompt_eval_count"] + response[1]["eval_count"]
    print(f"Prompt: {prompt}\nAnswer: {answer}")
    print(f"Tokens consumed : {tokens}\n")
