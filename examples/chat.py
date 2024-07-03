from dataformer.llms.openllm import OpenLLM
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("dataformer/self-knowledge")
datasetsub = dataset["train"].select(range(5))
instructions = [example["question"] for example in datasetsub]


request_list = [
    {"messages": [{"role": "user", "content": prompt}]} for prompt in instructions
]

"""
API Providers
- openai
- groq
- together
- anyscale
- deepinfra
- openrouter
"""

llm = OpenLLM(api_provider="openrouter")
response_list = llm.generate(request_list)

for request, response in zip(request_list, response_list):
    prompt = request["messages"][0]["content"]
    answer = response[-1]["choices"][0]["message"]["content"]
    print(f"Prompt: {prompt}\nAnswer: {answer}\n")
