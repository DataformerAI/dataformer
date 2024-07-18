from dataformer.llms.asyncllm import AsyncLLM

# Enter Ollama local or runpod URL
url = "https://a42o6nb8qoh1n6-11434.proxy.runpod.net/v1/chat/completions"  # OpenAI compatible API
url = "https://a42o6nb8qoh1n6-11434.proxy.runpod.net/api/chat"  # Ollama default API
model = "tinyllama"

llm = AsyncLLM(base_url=url, api_provider="ollama", model=model)

prompt = "hi tinyllama"
# Add stream = False in request
request_list = [{"messages": [{"role": "user", "content": prompt}], "stream": False}]


response_list = llm.generate(request_list)

for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[1]["choices"][0]["message"]["content"]
    print(f"Prompt: {prompt}\nAnswer: {answer}")
